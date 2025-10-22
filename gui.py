import ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage
from langchain.chat_models.base import BaseChatModel
from typing import Optional, List
from IntModel import *
import os
from tqdm import tqdm
from langchain.tools import StructuredTool
import pandas as pd
import sys

# Enforce ollama to check for GPU first
os.environ["OLLAMA_USE_GPU"] = "true"

# Example Functions
def escalate_to_hr(reason: str) -> str:
    return f"HR has been notified. Reason: {reason}"

def schedule_meeting_with_supervisor(topic: str) -> str:
    return f"Meeting with supervisor scheduled to discuss: {topic}"

def send_thank_you_email(message: str) -> str:
    return f"Sent thank-you email with message: {message}"

# Create StructuredTools from the functions
tools = [
    StructuredTool.from_function(
        func=escalate_to_hr,
        name="escalate_to_hr",
        description="Use this for serious or recurring complaints about management or workload."
    ),
    StructuredTool.from_function(
        func=schedule_meeting_with_supervisor,
        name="schedule_meeting_with_supervisor",
        description="Use this for workflow, scheduling, or communication issues."
    ),
    StructuredTool.from_function(
        func=send_thank_you_email,
        name="send_thank_you_email",
        description="Use this when feedback is positive or complimentary."
    )
]

def unwrap_output(result):
    """Extract content text if result is a LangChain message or result object."""
    try:
        # If it's an AIMessage
        if hasattr(result, "content"):
            return result.content
        # If it's a ChatResult or ChatGeneration
        if hasattr(result, "generations"):
            return result.generations[0].message.content
        # Otherwise just return the object itself
        return str(result)
    except Exception:
        return str(result)

import re

def maybe_run_tool(tools, response_text: str) -> str:
    """
    Detects the tool to call from the agent's response and executes it.
    Expects text like: 'Decision: **tool_name**'
    """
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # 1. Try to extract the decision (e.g., send_thank_you_email)
    match = re.search(r"Decision:\s*\*\*(\w+)\*\*", response_text)
    if not match:
        return "No decision found in model output."

    tool_name = match.group(1).strip()

    # 2. Find the corresponding tool
    for tool in tools:
        if tool.name.lower() == tool_name.lower():
            # Extract reasoning or context as optional argument
            # (the text after the Decision line)
            reasoning_match = re.split(r"Decision:\s*\*\*.*?\*\*\s*", response_text, maxsplit=1)
            reasoning = reasoning_match[1].strip() if len(reasoning_match) > 1 else ""

            try:
                result = tool.func(reasoning)
                return f"Executed `{tool.name}` successfully. Result: {result}"
            except Exception as e:
                return f"Tool `{tool.name}` failed: {e}"

    return f"Tool `{tool_name}` not found among available tools."

def extract_fields(pred_label):
    aspect_terms, aspect_categories, polarities, opinion_terms = [], [], [], []

    for d in pred_label:
        # If it's a Pydantic object
        if hasattr(d, "aspect_term"):
            aspect_terms.append(d.aspect_term)
            aspect_categories.append(d.aspect_category.value)
            polarities.append(d.polarity)
            opinion_terms.append(d.opinion_term)
        else:
            # If it's a string like "Aspect(aspect_term='SBAR', ...)"
            match = re.findall(
                r"aspect_term='([^']+)'.*?aspect_category=<[^:]+: '([^']+)'>.*?polarity='([^']+)'.*?opinion_term='([^']+)'",
                d
            )
            if match:
                term, category, polarity, opinion = match[0]
                aspect_terms.append(term)
                aspect_categories.append(category)
                polarities.append(polarity)
                opinion_terms.append(opinion)
    return aspect_terms, aspect_categories, polarities, opinion_terms


class OllamaLLM(BaseChatModel):
    model: str

    def _generate(self, messages: List[dict], stop: Optional[List[str]] = None, **kwargs: any):
        """Handle message-based input for LangChain's chat agent pipeline."""
        formatted_messages = []
        for m in messages:
            if isinstance(m, dict):
                formatted_messages.append(m)
            else:
                formatted_messages.append({
                    "role": getattr(m, "role", "user"),
                    "content": getattr(m, "content", str(m))
                })

        response = ollama.chat(model=self.model, messages=formatted_messages)
        content = response["message"]["content"]

        return self._create_chat_result(content)

    def _create_chat_result(self, content: str):
        from langchain.schema import ChatGeneration, ChatResult
        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"


    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "ollama"
    
     # Needed for tool calling
    def bind_tools(self, tools: list[any], **kwargs):
        """
        Simulate binding tools by returning self.
        This allows LangChain's create_tool_calling_agent to work even though
        Ollama does not natively support tool/function calling.
        """
        return self

# Load Data excluding the assignments
file_path = "datasets/asqp/nurse/test.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines_array = [
        line.split("#", 1)[0].strip()   # take only text before '#'
        for line in f.readlines()
        if line.split("#", 1)[0].strip()  # keep only non-empty pre-# parts
    ]

def run_predictions(model_name):
    # Define Components for First Chain
    llm = OllamaLLM(model=model_name)
    parser = PydanticOutputParser(pydantic_object=ASQPRecord)

    prompt_template = """
    You are provided the following nurse's comment:
    {input_text}

    Recognize all sentiment elements with their corresponding aspect terms, aspect categories,
    sentiment polarity, and opinion terms in the given schema.

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    agent1_chain = prompt | llm | parser

    prompt2 = PromptTemplate(
        template=(
            "You are a nurse manager reading both the nurse's comment and an extracted analysis of sentiments.\n\n"
            "Original nurse comment:\n{raw_input}\n\n"
            "Extracted sentiment analysis:\n{predicted_sentiments}\n\n"
            "Based on this, write a short, specific suggestion for improving the nurse's work environment. "
            "Output only plain text (no JSON, no markdown)."
        ),
        input_variables=["raw_input", "predicted_sentiments"],
    )

    llm2 = OllamaLLM(model=model_name)
    agent2_chain = prompt2 | llm2

    # Create a list to hold all prediction dicts
    records = []

    for line in tqdm(lines_array, desc="Processing items"):
        try:
            input_text = line.strip()

            asqp_result: ASQPRecord = agent1_chain.invoke({"input_text": input_text})

            aspect_terms, aspect_categories, polarities, opinion_terms = extract_fields(asqp_result.pred_label)

            rec = {
                "model": model_name,
                "raw_text": input_text,
                "aspect_terms": "; ".join(aspect_terms),
                "aspect_categories": "; ".join(aspect_categories),
                "polarities": "; ".join(polarities),
                "opinion_terms": "; ".join(opinion_terms)
            }

            records.append(rec)

        except Exception as e:
            records.append({
                "model": model_name,
                "raw_text": line.strip(),
                "aspect_terms": [],
                "aspect_categories": [],
                "polarities": [],
                "opinion_terms": [],
                "error": str(e)
            })

    # ✅ Convert to Pandas DataFrame
    df = pd.DataFrame(records)

    return df

model_type = "gemma3:4b"
dir_path = "generations/pipeline"
output_file = f"{dir_path}/{model_type.split(':')[0]}_0.csv"

if not os.path.exists(output_file):
    df = run_predictions(model_type)
    df.to_csv(f'{output_file}', index=False)
else:
    df = pd.read_csv(output_file)

import pandas as pd
from tkinter import Tk, Label, Entry, Button, Text, Scrollbar, Frame, END

pending_tool_decision = None  # Global variable to store proposed action

# ---- Compute summary statistics for the manager ----
def get_polarity_counts(df):
    counts = {}
    for row in df['polarities']:
        overall_polarity = {}
        if not row:
            continue
        for polarity in row.split("; "):
            overall_polarity[polarity] = overall_polarity.get(polarity, 0) + 1
        max_polarity = max(overall_polarity, key=overall_polarity.get)
        counts[max_polarity] = counts.get(max_polarity, 0) + 1
    return counts

def get_category_counts(df):
    counts = {}
    for row in df['aspect_categories']:
        overall_category = {}
        if not row:
            continue
        for category in row.split("; "):
            overall_category[category] = overall_category.get(category, 0) + 1
        max_category = max(overall_category, key=overall_category.get)
        counts[max_category] = counts.get(max_category, 0) + 1
    return counts

polarity_counts = get_polarity_counts(df)
category_counts = get_category_counts(df)

# ---- Build manager chatbot GUI ----
chat_context = f"""
You are an assistant for a nurse manager. 
You have access to the following nurse feedback dataset (showing only the first 10 records for context):
{df.head(10).to_dict(orient='records')}

Summary statistics:
- Polarity counts: {polarity_counts}
- Category counts: {category_counts}

You can answer questions about:
- sentiments
- aspect categories
- recommendations
- counts/statistics per category
- which actions/tools might be appropriate for certain issues

TOOLS AVAILABLE:
{', '.join([t.name for t in tools])}

Instructions for taking action:
- If the manager approves an action, you must output exactly one line starting with:
  Decision: **<tool_name>**
  where <tool_name> matches one of the available tools (case-insensitive).
- Any text following the tool name will be used as the argument for that tool (reasoning or context).
- Do NOT use angle brackets <> or other punctuation in the tool name.
- Only output this line if an action is recommended; otherwise explain why no action is needed in plain text.
- Example of correct format:
  Decision: **escalate_to_hr** Nurse reported repeated workload issues.
- If no action is needed, output a clear explanation in plain text without the "Decision:" line.

Always follow these formatting rules strictly so your output can be parsed by the system.
"""

# -----------------------------
# Tkinter GUI
# -----------------------------
root = Tk()
root.title("Nurse Manager Chat")
root.geometry("1000x700")

# Configure grid weights for expansion
root.grid_rowconfigure(1, weight=1)  # Chat area
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# -----------------------------
# Stats Panel
# -----------------------------
stats_frame = Frame(root, bd=2, relief='sunken')
stats_frame.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=10, pady=5)

stats_text = Text(stats_frame, height=8, state='disabled', wrap='word', font=("Arial", 11))
stats_scroll = Scrollbar(stats_frame, command=stats_text.yview)
stats_text.configure(yscrollcommand=stats_scroll.set)

stats_text.pack(side='left', fill='both', expand=True)
stats_scroll.pack(side='right', fill='y')

def refresh_stats():
    stats_text.config(state='normal')
    stats_text.delete(1.0, END)
    stats_text.insert(END, "=== Nurse Feedback Statistics ===\n")
    stats_text.insert(END, f"Polarity counts: {polarity_counts}\n")
    stats_text.insert(END, f"Category counts: {category_counts}\n\n")
    stats_text.insert(END, "Top 5 entries:\n")
    for _, row in df.head(5).iterrows():
        stats_text.insert(END, f"- {row['raw_text']} [{row['polarities']}] ({row['aspect_categories']})\n")
    stats_text.config(state='disabled')

refresh_stats()

def clear_chat():
    chat_text.config(state='normal')
    chat_text.delete('1.0', END)
    chat_text.config(state='disabled')

# -----------------------------
# Chat Area
# -----------------------------
chat_text = Text(root, state='disabled', wrap='word', font=("Arial", 12))
chat_scroll = Scrollbar(root, command=chat_text.yview)
chat_text.configure(yscrollcommand=chat_scroll.set)

chat_text.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=5)
chat_scroll.grid(row=1, column=2, sticky='ns', pady=5)

# -----------------------------
# User Input
# -----------------------------
user_input = Entry(root, font=("Arial", 12))
user_input.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=5)

send_btn = Button(root, text="Send", command=lambda: send_message())
send_btn.grid(row=2, column=2, padx=5, pady=5)

approve_btn = Button(root, text="Approve Action", command=lambda: approve_action())
approve_btn.grid(row=3, column=0, sticky='ew', padx=10, pady=5)

reject_btn = Button(root, text="Reject Action", command=lambda: reject_action())
reject_btn.grid(row=3, column=1, sticky='ew', padx=10, pady=5)

clear_btn = Button(root, text="Clear Chat", command=clear_chat)
clear_btn.grid(row=3, column=2, padx=10, pady=5)

# -----------------------------
# Chat Functions
# -----------------------------
pending_tool_decision = None

def query_manager_chat(user_input):
    llm3 = OllamaLLM(model=model_type)  # reuse your existing Ollama model
    prompt = f"{chat_context}\nManager: {user_input}\nAssistant:"
    response = llm3._generate([{"role": "user", "content": prompt}])
    return unwrap_output(response)

def send_message():
    global pending_tool_decision
    msg = user_input.get().strip()
    if not msg:
        return

    chat_text.config(state='normal')
    chat_text.insert(END, f"You: {msg}\n\n")

    response = query_manager_chat(msg)
    chat_text.insert(END, f"NurseBot: {response}\n\n")
    chat_text.config(state='disabled')
    chat_text.see(END)
    user_input.delete(0, END)

    # Detect tool decision
    match = re.search(r"Decision:\s*\*\*(\w+)\*\*\s*(.*)", response)
    if match:
        tool_name, reasoning = match.group(1), match.group(2).strip()
        pending_tool_decision = (tool_name, reasoning)

        chat_text.config(state='normal')
        chat_text.insert(END, f"Pending Action: {tool_name} - '{reasoning}'\n")
        chat_text.insert(END, f"Manager: Click 'Approve' to execute or 'Reject' to cancel.\n\n")
        chat_text.config(state='disabled')
        chat_text.see(END)

def approve_action():
    global pending_tool_decision
    if pending_tool_decision:
        tool_name, reasoning = pending_tool_decision
        tool_response = maybe_run_tool(tools, f"Decision: **{tool_name}** {reasoning}")

        chat_text.config(state='normal')
        chat_text.insert(END, f"Tool Execution Result: {tool_response}\n\n")
        chat_text.config(state='disabled')
        chat_text.see(END)
        pending_tool_decision = None
    else:
        chat_text.config(state='normal')
        chat_text.insert(END, "No pending action to approve.\n\n")
        chat_text.config(state='disabled')
        chat_text.see(END)

def reject_action():
    global pending_tool_decision
    if pending_tool_decision:
        chat_text.config(state='normal')
        chat_text.insert(END, f"Action '{pending_tool_decision[0]}' was rejected by manager.\n\n")
        chat_text.config(state='disabled')
        chat_text.see(END)
        pending_tool_decision = None
    else:
        chat_text.config(state='normal')
        chat_text.insert(END, "No pending action to reject.\n\n")
        chat_text.config(state='disabled')
        chat_text.see(END)

# -----------------------------
# Start GUI
# -----------------------------
root.mainloop()