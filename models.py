import os
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import logging  
  
# Configure logging  
logging.basicConfig(level=logging.INFO)  
  
# Error callback function
def log_retry_error(retry_state):  
    logging.error(f"Retrying due to error: {retry_state.outcome.exception()}")  

DEFAULT_CONFIG = {
    "engine": None,
    "model": 'Mistral-7B-Instruct-v0.2',
    #"model": "Llama-2-7b-chat-hf",
    #"model": "Mistral-7B-Instruct-v0.2",
    "temperature": 0.0,
    "max_tokens": 5000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

class OpenAIWrapper:
    def __init__(self, config = DEFAULT_CONFIG, system_message=""):
        # TODO: set up your API key with the environment variable OPENAIKEY
        openai.api_key = os.environ.get("OPENAI_API_KEY")      
        openai.api_key = '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'
        if os.environ.get("USE_AZURE")=="True":
            print("using azure api")
            openai.api_type = "azure"
        # openai.api_base = os.environ.get("API_BASE")
        openai.api_base = "http://localhost:8000/v1"
        openai.api_version = os.environ.get("API_VERSION")

        self.config = config
        print("api config:", config, '\n')

        # count total tokens
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # system message
        self.system_message = system_message # "You are an AI assistant that helps people find information."

    # retry using tenacity
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry_error_callback=log_retry_error)
    def completions_with_backoff(self, **kwargs):
        # print("making api call:", kwargs)
        # print("====================================")
        return openai.ChatCompletion.create(**kwargs)

    def run(self, prompt, n=1, system_message=""):
        """
            prompt: str
            n: int, total number of generations specified
        """
        try:
            # overload system message
            if system_message != "":
                sys_m = system_message
            else:
                sys_m = self.system_message
            if sys_m == "":
                print("adding system message:", sys_m)
                # print("user content:", prompt)
                messages = [
                    {"role":"system", "content":sys_m},
                    {"role":"user", "content":prompt}# 此处的prompt来自基础模型的指令
                ]
            else:
                messages = [
                    {"role":"user","content":prompt}
                ]
            text_outputs = []
            raw_responses = []
            while n > 0:
                cnt = min(n, 5) # number of generations per api call
                n -= cnt
                #使用基础模型
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,6,7"
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                #os.environ["TORCH_USE_CUDA_DSA"] = "1"
                device = "cuda" # the device to load the model onto
                # device_map = {'cpu': 'cpu', 'cuda:0': 'cuda:0', 'cuda:1': 'cuda:1', 'cuda:2': 'cuda:2', 'cuda:3': 'cuda:3'}
                model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map='auto' if torch.cuda.is_available() else None)
                tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

                model_inputs = encodeds.to(device)
                # model.to(device)

                generated_ids, agent_string = model.generate(model_inputs, max_new_tokens=10000, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
            return decoded, agent_string
        except Exception as e:
            print("an error occurred:", e)
            return [], []

    def compute_gpt_usage(self):
        engine = self.config["engine"]
        if engine == "devgpt4-32k":
            cost = self.completion_tokens / 1000 * 0.12 + self.prompt_tokens / 1000 * 0.06
        else:
            cost = 0 # TODO: add custom cost calculation for other engines
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}