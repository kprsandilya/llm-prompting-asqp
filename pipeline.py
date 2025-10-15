import ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from typing import Optional, List
from IntModel import *
import os
import json
from tqdm import tqdm
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor

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



def build_tool_agent(model_name: str, tools):
    llm3 = OllamaLLM(model=model_name)

    system_prompt = (
        "You are an assistant for a nurse manager. "
        "Given a workplace suggestion, decide whether action is needed and use one of the available tools below. "
        "Format the chosen function as Decision: **<chosen_function>** "
        "If no action is necessary, explain why.\n\n"
        "TOOLS AVAILABLE:\n"
        f"{', '.join([t.name for t in tools])}"
    )

    # ✅ Add `agent_scratchpad`
    prompt3 = PromptTemplate(
        template=(
            "{system_prompt}\n\n"
            "Suggestion:\n{suggestion}\n\n"
            "{agent_scratchpad}"
        ),
        input_variables=["suggestion", "agent_scratchpad"],
        partial_variables={"system_prompt": system_prompt}
    )

    agent = create_tool_calling_agent(llm3, tools, prompt3)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return executor


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

# ---- Load data ----
file_path = "datasets/asqp/nurse/test.txt"
with open(file_path, "r", encoding='utf-8') as f:
    lines_array = f.readlines()

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

    # Build Chain 1
    agent1_chain = prompt | llm | parser

    # Define Componets for the Second Chain
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

    # Build Chain 2
    agent2_chain = prompt2 | llm2

    # Tooling Agent
    executor = build_tool_agent(model_name, tools)

    # Array of JSONs for output
    output_jsons = []

    for line in tqdm(lines_array[:5], desc="Processing items"):

        try:
            # ---- Run pipeline ----
            input_text = line

            asqp_result: ASQPRecord = agent1_chain.invoke({"input_text": input_text})

            manager_result = agent2_chain.invoke({
                "raw_input": input_text,
                "predicted_sentiments": asqp_result.model_dump_json(indent=2)
            })

            tool_response = executor.invoke({"suggestion": manager_result})
            tool_execution = maybe_run_tool(tools, unwrap_output(tool_response))

            prediction = { 
                "model": model_name,
                "raw_text": line,
                "pred_label": asqp_result.model_dump(),
                "recommendation": unwrap_output(manager_result),
                "tool_decision": unwrap_output(tool_response),
                "tool_result": tool_execution
            }

            output_jsons.append(dict(prediction))
        except Exception as e:
            prediction = { 
                "model": model_name,
                "raw_text": line,
                "pred_label": {},
                "recommendation": "Failure",
                "error": str(e)
            }
            output_jsons.append(dict(prediction))

    return output_jsons

model_type = "gemma3:4b"
dir_path = "generations/pipeline"
output_file = f"{dir_path}/{model_type.split(':')[0]}_0.json"

if not os.path.exists(output_file):
    output_jsons = run_predictions(model_type)
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_jsons, json_file, ensure_ascii=False, indent=4)
else:
    print(f"{output_file} already exists!")