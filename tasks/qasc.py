import os
import re
from tasks.base import Task, DATA_PATH
from prompts.qasc import standard_prompt, cot_prompt, spp_prompt
import json
# from models import gpt

# def extract_and_join_numbers(s):
#     # 匹配以负号开始的数字序列，或者独立的数字序列
#     pattern = r'-?\d+'
#     matches = re.findall(pattern, s)
#     # 使用空格将匹配的字符串拼接起来
#     return ' '.join(matches)

def remove_punctuation(output: str) -> str:
    markers = [",", ";", ":", ".", '"']
    for marker in markers:
        output = output.replace(marker, "")
    return output

def convert_newline_to_space(output: str) -> str:
    output = output.replace("\n", " ")
    return output

def eval_for_exact_matching_with_no_punctuation(
    input: str, output: str, target: str
) -> bool:
    output = remove_punctuation(output)
    output = convert_newline_to_space(output)
    if target in output:
        return True
    return False
    
class qascTask(Task):
    def __init__(self, file='qasc.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'qasc', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        task = 'Fact1:' + datapoint['fact1'] + '\n' + 'Fact2:' + datapoint['fact2'] + '\n' + 'Question:' + datapoint['formatted_question']
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
        target = '(' + instance["answerKey"] + ')'
        flag = eval_for_exact_matching_with_no_punctuation(input, output, target)
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
                if len(response.split("Final answer:")) >= 3:
                    return response.split("Final answer:")[2].strip(), True
                else:
                    return response, False
            elif "final answer:" in response:
                if len(response.split("final answer:")) >= 2:
                    return response.split("final answer:")[1].strip(), True
                else:
                    return response, False
            if "Assistant:" in response:
                if len(response.split("Assistant: ")) >= 2:
                    return response.split("Assistant: ")[1].strip(), True
            return response, True
        
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