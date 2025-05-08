import os
import random

class PromptLoader:
    def __init__(self, base_path="prompt"):
        self.base_path = base_path   
        
    def load_prompt(self, task="asqp", prediction_type="label", aspects=[], examples=[], seed_examples=42, input_example=None, polarities=["positive", "negative", "neutral"], reasoning_step="reasoning", shuffle_examples=True, load_llm_instruction=False, language="en"):
        if load_llm_instruction:
           # Load Base Prompt
           prompt_path = os.path.join(self.base_path, task, f"llm_instruction.txt")
           if not os.path.exists(prompt_path):
               raise FileNotFoundError(f"Prompt file {prompt_path} not found.")

           with open(prompt_path, 'r', encoding='utf-8') as file:
               prompt = file.read()        
           return prompt  
         
        # random shuffle examples with seed
        random.seed(seed_examples)
        
        if shuffle_examples:
           random.shuffle(examples)
        
        # Load Base Prompt
        if language != "en":
            prompt_path = os.path.join(self.base_path, task, f"prompt_{language}.txt")
        else:
           prompt_path = os.path.join(self.base_path, task, f"prompt.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file {prompt_path} not found.")

        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
            
        # Set Prompt Header
        prompt = prompt.replace("[[aspect_category]]", str(aspects)[1:-1])

        # Set Examples in Prompt
        if len(examples) == 0:
            prompt = prompt.replace("[[examples]]", "")
        elif prediction_type in ["label", "chain-of-thought", "plan-and-solve"]:
            example_str = ""
            for example in examples:
                example_str += f'Text: {example["text"]}\nSentiment Elements: {str(example["tuple_list"])}\n'        
            prompt = prompt.replace("[[examples]]", example_str)
            
        elif prediction_type == "label-and-chain-of-thought":
            example_str = ""
            for example in examples:
                example_str += f'Text: {example["text"]}\nReasoning: {example["pred_reasoning"]}\nSentiment Elements: {str(example["tuple_list"])}\n'
            prompt = prompt.replace("[[examples]]", example_str)
            
            
        # Set Footer of Example to be predicted
        if prediction_type == "label":
           prompt += "Text: " + input_example["text"] + "\nSentiment Elements:"
            
        if prediction_type == "plan-and-solve":
           prompt += "Text: " + input_example["text"] + "\n\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."

        if prediction_type == "chain-of-thought":
           prompt += "Text: " + input_example["text"] + "\n\nLet's think think step by step."

        return prompt