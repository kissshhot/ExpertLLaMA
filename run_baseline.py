from template import template_gen_expert_identity, template_expert_prompting

# import logging
# # 配置日志记录器
# logging.basicConfig(filename='/home/dyf/SPP_test/SPP5_way1_no_sample_more/logs/history.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import json
import argparse
from models import OpenAIWrapper
from tasks import get_task
import time
from tqdm import tqdm
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModel
# from transformers import set_seed
# seed = 10
# set_seed(seed=seed)
from prompts.role_describe import historyExpert1, storyWriter1, historyExpert2, historyExpert3, storyWriter2, storyWriter3, advisor1, advisor2, advisor3,engineer1,engineer2,engineer3,entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
model = AutoModelForCausalLM.from_pretrained("/data1/dyf/model/agent_qwen/", device_map = 'auto' if torch.cuda.is_available() else None)
tokenizer = AutoTokenizer.from_pretrained("/data1/dyf/model/agent_qwen/", use_fast='store_true') # ,use_fast='store_true'
# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = 32000

import numpy as np

SLEEP_RATE = 10 # sleep between calls
def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def _post_process_raw_response(task, raw_output_batch, method):
    unwrapped_output_batch = []
    if_success_batch = []
    for output in raw_output_batch:
        unwrapped_output, if_success_flag = task.prompt_unwrap(output, method)
        unwrapped_output_batch.append(unwrapped_output)
        if_success_batch.append(if_success_flag)
    return unwrapped_output_batch, if_success_batch

def _run_task(task_name, gpt, task, i, method, num_generation, n, th, T):
    if task_name in ['trivia_creative_writing', 'logic_grid_puzzle']:
        prompt = task.get_input_prompt(i, method='Expertllama_expert_identity')
        messages = [
            {"role":"system","content":''},
            {"role":"user","content":prompt}
        ]
        device = "cuda"
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        # test = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        # model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_output_batch = decoded
        expert_identity = raw_output_batch[0].split('Agent Description:')[2].replace("\n", "")
        expert_identity = expert_identity[:-10]
        # time.sleep(5)
        if raw_output_batch == []: # handle exception
            return {}
        
        prompt = task.get_input_prompt_Expertllama(i, expert_identity = expert_identity)
        messages = [
            {"role":"system","content":''},
            {"role":"user","content":prompt}
        ]
        device = "cuda"
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_output_batch = decoded
        unwrapped_output_batch, if_success_batch = _post_process_raw_response(task, raw_output_batch, method)
        # compute automatic metric (different for each task), e.g., if the output contains all the answers
        test_output_infos = [task.test_output(i, output) for output in unwrapped_output_batch]
        # log output
        log_output = {
            "idx": i,
            "unwrapped_output": unwrapped_output_batch,
            "parsing_success_flag": if_success_batch,
            "test_output_infos": test_output_infos
        }
    elif task_name == 'codenames_collaborative':
        spymaster_prompt = task.get_input_prompt(i, method=method, role='spymaster')
        messages = [
            {"role":"system","content":''},
            {"role":"user","content":spymaster_prompt}
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to('cuda')
        # model.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_spymaster_output = decoded
        expert_identity = raw_spymaster_output[0].split('Agent Description:')[2].replace("\n", "")
        expert_identity = expert_identity[:-10]
        # time.sleep(5)
        if raw_spymaster_output == []: # handle exception
            return {}
        
        prompt = task.get_input_prompt_Expertllama_spymaster(i, expert_identity = expert_identity, role='spymaster')
        messages = [
            {"role":"system","content":''},
            {"role":"user","content":prompt}
        ]
        device = "cuda"
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        # model_text.eval()
        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_spymaster_output[0] = raw_spymaster_output[0][:-10]
        spymaster_output, if_success_batch_spymaster = _post_process_raw_response(task, raw_spymaster_output, method)
        hint_word = spymaster_output[0].replace(".", "").strip()
        print(f"\tidx: {i} | done spymaster, hint word: {hint_word}")

        # sleep before calling guesser
        time.sleep(SLEEP_RATE)
        # get guesser result
        guesser_prompt = task.get_input_prompt(i, method=method, role='guesser', hint_word=hint_word)
        messages1 = [
            {"role":"system","content":''},
            {"role":"user","content":guesser_prompt}
        ]
        encodeds = tokenizer.apply_chat_template(messages1, return_tensors="pt")

        model_inputs = encodeds.to('cuda')

        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_guesser_output = decoded
        expert_identity = raw_guesser_output[0].split('Agent Description:')[2].replace("\n", "")
        expert_identity = expert_identity[:-10]
        # time.sleep(5)
        if raw_guesser_output == []: # handle exception
            return {}
        
        prompt = task.get_input_prompt_Expertllama_guesser(i, expert_identity = expert_identity, role='guesser', hint_word=hint_word)
        messages = [
            {"role":"system","content":''},
            {"role":"user","content":prompt}
        ]
        device = "cuda"
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
        decoded = tokenizer.batch_decode(generated_ids)
        raw_guesser_output[0] = raw_guesser_output[0][:-10]
        guesser_output_batch, if_success_batch_guesser = _post_process_raw_response(task, raw_guesser_output, method)
        # compute automatic metric (different for each task), e.g., if the output contains all the answers
        test_output_infos = [task.test_output(i, output) for output in guesser_output_batch]
        # log output
        log_output = {
            "idx": i,
            "spymaster_output": spymaster_output,
            "guesser_output": guesser_output_batch,
            "hint_word": hint_word,
            "parsing_success_flag_spymaster": if_success_batch_spymaster,
            "parsing_success_flag_guesser": if_success_batch_guesser,
            "test_output_infos": test_output_infos,
            "raw_spymaster_output": raw_spymaster_output,
            "raw_guesser_output": raw_guesser_output
        }
    else:
        raise NotImplementedError(f"task {task_name} not implemented; please choose from ['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative']")
    #一个任务完成了，此时需要把上一次的任务文件清除
    # 文件路径
    file_path = 'agent_string.pkl'
    # 删除文件
    if os.path.exists(file_path):
        os.remove(file_path)
        print("文件已成功删除")
    else:
        print("文件不存在，无法删除")
    file_path = 'agent_tensor.pkl'
    # 删除文件
    if os.path.exists(file_path):
        os.remove(file_path)
        print("文件已成功删除")
    else:
        print("文件不存在，无法删除")
    # log everything else that is related
    log_output.update(args)
    log_output.update({"task_data":task.get_input(i)})
    return log_output

def run(args):
    # get configs
    gpt_config = args['gpt_config']
    task_name = args['task']
    method = args['method']
    start_idx, end_idx = args['task_start_index'], args['task_end_index']
    task_data_file = args['task_data_file']
    num_generation = args['num_generation']
    n = args['n']
    th = args['th']
    T = args['T']

    additional_output_note = args['additional_output_note']
    system_message = args['system_message']
    print(f"setting default system message: {system_message}")
    
    # setup gpt api
    gpt = OpenAIWrapper(config=gpt_config, system_message=system_message)

    # setup log file
    if system_message == "":
        log_file = f"logs/{task_name}/{th}_{n}_{T}_{start_idx}_{end_idx}.jsonl"
    else:
        log_file = f"logs/{task_name}/{th}_{n}_{T}_{start_idx}_{end_idx}.jsonl"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # setup task
    task = get_task(task_name, file=task_data_file)
    
    all_logs = []
    print("start running ... log file:", log_file)
    print('len(task)', len(task))
    print()
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    print("total num of instances:", end - start)
    for i in tqdm(range(start, end)):
        log_output = _run_task(task_name, gpt, task, i, method, num_generation, n, th, T)
        all_logs.append(log_output)
        print("\tidx:", i, "done | usage so far:", gpt.compute_gpt_usage())
        # output log at each iteration
        output_log_jsonl(log_file, all_logs)
        # sleep
        time.sleep(SLEEP_RATE)



# TODO: add your custom model config here:
gpt_configs = {
    #Llama-2-13b-chat-hf
    #vicuna-7b-v1.5
    #Mistral-7B-Instruct-v0.2
    "Mistral-7B-Instruct-v0.2": {
        "engine": None,
        "model": "Mistral-7B-Instruct-v0.2",
        #"model": "Llama-2-7b-chat-hf",
        #"model": "Mistral-7B-Instruct-v0.2",
        "temperature": 0.0,
        "max_tokens": 5000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    }
}

default_gpt_config = {
    "engine": None,
    "temperature": 0.0,
    "max_tokens": 5000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

def parse_args():
    model_choices = list(gpt_configs.keys())
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='Mistral-7B-Instruct-v0.2', choices=model_choices)
    args.add_argument('--n', type=int, default=3, choices=[1,2,3,4])
    args.add_argument('--T', type=float, default=1.0, choices=[0.1,0.2,0.5,1,2,5,10])
    args.add_argument('--th', type=float, default=0.5, choices=[0.4,0.45,0.5,0.55,0.6,0.65,0.7,2.0])
    args.add_argument('--method', type=str, default='spp', choices=['standard','cot','spp','spp_profile', 'spp_fixed_persona'])
    args.add_argument('--task', type=str, default='trivia_creative_writing', choices=['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative'])
    args.add_argument('--task_data_file', type=str, default='trivia_creative_writing_100_n_5.jsonl')# logic_grid_puzzle_200.jsonl trivia_creative_writing_100_n_5.jsonl codenames_50.jsonl
    args.add_argument('--task_start_index', type=int, default=1)#4 55 57
    args.add_argument('--task_end_index', type=int, default=2)
    args.add_argument('--num_generation', type=int, default=1)
    args.add_argument('--additional_output_note', type=str, default="")
    args.add_argument('--temperature', type=float, default=0.0)
    args.add_argument('--top_p', type=float, default=1.0)
    args.add_argument('--system_message', type=str, default="You are an AI assistant that helps people find information.")
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = vars(parse_args())
    model_name = args['model']
    
    if model_name in gpt_configs:
        args['gpt_config'] = gpt_configs[model_name] # our configs
    else:
        args['gpt_config'] = default_gpt_config
        args['gpt_config']['engine'] = model_name
    
    # overwrite temperature and top_p
    args['gpt_config']['temperature'] = args['temperature']
    args['gpt_config']['top_p'] = args['top_p']
    print("run args:", args)
    
    run(args)