import os
import re
from tasks.base import Task, DATA_PATH
from prompts.sonnets import standard_prompt, cot_prompt, spp_prompt
from .sonnet_eval import sonnet_errors
import json
# from models import gpt

def eval_for_Sonnet(output: str, rhyme_scheme: str) -> bool:
    try:
        errors = sonnet_errors(output, rhyme_scheme)
        if not errors:
            return True
        return False
    except Exception as e:
        return False
    
class sonnetsTask(Task):
    def __init__(self, file='sonnets.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'sonnets', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        task = datapoint["input"]
        # task_str = " ".join(task)
        
        if method == "standard":
            input_prompt = standard_prompt.format(task=task)
        elif method == "cot":
            input_prompt = cot_prompt.format(task=task)
        elif method == "spp":
            input_prompt = spp_prompt.format(task=task)
        # elif method == "spp_profile":
        #     input_prompt = spp_prompt_profile.format(task=task)
        else:
            raise NotImplementedError(f"method {method} not implemented")
        
        return input_prompt
        
    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia task
        instance = self.data[idx]
        flag = eval_for_Sonnet(output, "ABAB CDCD EFEF GG")
        info = {'correct_flag': flag}
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        if method == "standard":
            if "**Final answer**:" in response:
                if len(response.split("**Final answer**:")) >= 2:
                    return response.split("**Final answer**:")[1].strip(), True
                else:
                    return response, False
            if "Final Answer:" in response:
                if len(response.split("Final Answer:")) >= 2:
                    return response.split("Final Answer:")[1].strip(), True
                else:
                    return response, False
            elif "Final answer:" in response:
                if len(response.split("Final answer:")) >= 5:
                    return response.split("Final answer:")[4].strip(), True
                else:
                    return response, False
            elif "final answer:" in response:
                if len(response.split("final answer:")) >= 5:
                    return response.split("final answer:")[4].strip(), True
                else:
                    return response, False
            else:
                return response, False
        
        elif method == "cot":
            if "**Final answer**:" in response:
                if len(response.split("**Final answer**:")) >= 2:
                    return response.split("**Final answer**:")[1].strip(), True
                else:
                    return response, False
            if "Final Answer:" in response:
                if len(response.split("Final Answer:")) >= 2:
                    return response.split("Final Answer:")[1].strip(), True
                else:
                    return response, False
            elif "Final answer:" in response:
                if len(response.split("Final answer:")) >= 5:
                    return response.split("Final answer:")[4].strip(), True
                else:
                    return response, False
            elif "final answer:" in response:
                if len(response.split("final answer:")) >= 5:
                    return response.split("final answer:")[4].strip(), True
                else:
                    return response, False
            else:
                return response, False
        
        elif method in ["spp","spp_profile","spp_fixed_persona"]:
            #Final answer更靠后 **Final answer**:
            if "**Final answer**:" in response:
                if len(response.split("**Final answer**:")) >= 2:
                    return response.split("**Final answer**:")[1].strip(), True
                else:
                    return response, False
            if "Final Answer:" in response:
                if len(response.split("Final Answer:")) >= 2:
                    return response.split("Final Answer:")[1].strip(), True
                else:
                    return response, False
            elif "Final answer:" in response:
                if len(response.split("Final answer:")) >= 5:
                    return response.split("Final answer:")[4].strip(), True
                else:
                    return response, False
            elif "final answer:" in response:
                if len(response.split("final answer:")) >= 5:
                    return response.split("final answer:")[4].strip(), True
                else:
                    return response, False
            else:
                return response, False
        
        else:
            raise NotImplementedError(f"method {method} not implemented")