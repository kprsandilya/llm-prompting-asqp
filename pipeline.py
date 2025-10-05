import ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms.base import LLM
from typing import Optional, List
from IntModel import *


# ---- Custom Ollama Wrapper ----
class OllamaLLM(LLM):
    """LangChain wrapper for the Ollama Python client."""
    model: str = "gemma3"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "ollama"


# ---- Load data ----
file_path = "datasets/asqp/nurse/test.txt"
with open(file_path, "r", encoding='utf-8') as f:
    lines_array = f.readlines()


# ---- Define components ----
llm = OllamaLLM(model="gemma3:4b")
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

# ---- Build modern LangChain pipeline ----
agent1_chain = prompt | llm | parser

# ---- Run pipeline ----
input_text = lines_array[0]
agent1_output: ASQPRecord = agent1_chain.invoke({"input_text": input_text})
print(agent1_output)

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

llm2 = OllamaLLM(model="gemma3:4b")

agent2_chain = prompt2 | llm2

asqp_result = agent1_chain.invoke({"input_text": input_text})

manager_result = agent2_chain.invoke({
    "raw_input": input_text,
    "predicted_sentiments": asqp_result.model_dump_json(indent=2)
})

print(manager_result)