import ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms.base import LLM
from typing import Optional, List
from IntModel import *
import os
import json
from tqdm import tqdm

# Enforce ollama to check for GPU first
os.environ["OLLAMA_USE_GPU"] = "true"

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

    # Array of JSONs for output
    output_jsons = []

    for line in tqdm(lines_array, desc="Processing items"):

        try:
            # ---- Run pipeline ----
            input_text = line

            asqp_result: ASQPRecord = agent1_chain.invoke({"input_text": input_text})

            manager_result = agent2_chain.invoke({
                "raw_input": input_text,
                "predicted_sentiments": asqp_result.model_dump_json(indent=2)
            })

            prediction = { 
                "model": model_name,
                "raw_text": line,
                "pred_label": asqp_result.model_dump(),
                "recommendation": manager_result
            }

            output_jsons.append(dict(prediction))
        except Exception as e:
            prediction = { 
                "model": model_name,
                "raw_text": line,
                "pred_label": {},
                "recommendation": "Failure",
                "error": e
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