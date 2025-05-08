import ollama
import time
import requests
from openai import OpenAI
import os

class LLM:
    def __init__(self, base_model="gemma3:27b", gwdg_token="", openai_token="",  parameters=[{"name": "stop", "value": ["\\n"]}, 
                                                                                  {"name": "num_ctx", "value": "8192"}]):
        self.model_name = base_model
        self.gwdg_token = gwdg_token
        self.openai_token = openai_token
        
        if self.openai_token != "":
            self.openai_client = OpenAI(api_key=self.openai_token)
            
        # if self.model_name == "qwen3:30b":
        #   from mlx_lm import load, generate
        #   self.mlx_model, self.mlx_tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")
    
    def predict_mlx(self, prompt, seed=0, stop=["]"], temperature=0.8, num_ctx=4096):
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        prediction_start_time = time.time()
        messages = [{"role": "user", "content": prompt}]
        prompt = self.mlx_tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        sampler = make_sampler(temperature)
        response = generate(self.mlx_model, self.mlx_tokenizer, prompt=prompt, verbose=False, sampler=sampler, num_ctx=num_ctx)
        duration = time.time() - prediction_start_time
        return response, duration
        
    def predict(self, prompt, seed=0, stop=["]"], temperature=0.8, num_ctx=4096):
        prediction_start_time = time.time()
        if (self.gwdg_token == "" and self.openai_token == ""):
            response_generated = False
            while not response_generated:
               response = ollama.generate(model=self.model_name, options=dict(seed=seed, temperature=temperature, num_ctx=num_ctx, stop=stop), prompt=prompt)
               response_generated = True
            response = response["response"] + "]"
        if (self.openai_token != ""):
          chat_completion = self.openai_client.chat.completions.create(messages=[{"role": "user", "content": prompt, "stop": ["]"]}],model=self.model_name)
          response = chat_completion.choices[0].message.content
          
          
        if (self.gwdg_token != ""):
          url = "https://chat-ai.academiccloud.de/v1/completions"

          headers = {
           "Accept": "application/json",
           "Authorization": f"Bearer {self.gwdg_token}",
           "Content-Type": "application/json"
          }
          
          data = {
              "model": self.model_name,
              "prompt": prompt,
              "max_tokens": 200,
              "temperature": 0.8,
              "stop": ["]"]
          }
          
          response = requests.post(url, json=data, headers=headers).json()["choices"][0]["text"] + "]"
          
        duration = time.time() - prediction_start_time
        return response, duration