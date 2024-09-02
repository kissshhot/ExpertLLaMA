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
from peft import PeftModel
#使用peftmodel加载lora
# lora_model_name_or_path = "/data1/dyf/LF_model/storyWriter/checkpoint-1700/"
lora_model_name_or_path = "/data1/dyf/agent_qwen_lora/storyWriter/checkpoint-1700/"
#### 初始化PeftModel, 并且load第一个adapter
lora_model = PeftModel.from_pretrained(model, model_id = lora_model_name_or_path, adapter_name = "storyWriter")
#### 读取另外两个adapter
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/advisor/checkpoint-1200/",adapter_name = "advisor")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/historyExpert/checkpoint-1100/",adapter_name = "historyExpert")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/engineer/checkpoint-900/", adapter_name = "engineer")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/entrepreneur/checkpoint-1400/", adapter_name = "entrepreneur")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/physician/checkpoint-1800/", adapter_name = "physician")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/psychologist/checkpoint-1200/", adapter_name = "psychologist")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/salesman/checkpoint-1000/", adapter_name = "salesman")
# lora_model.load_adapter(model_id = "/data1/dyf/LF_model/scientist/checkpoint-1900/", adapter_name = "scientist")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/advisor/checkpoint-900/",adapter_name = "advisor")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/historyExpert/checkpoint-800/",adapter_name = "historyExpert")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/engineer/checkpoint-1800/", adapter_name = "engineer")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/entrepreneur/checkpoint-1100/", adapter_name = "entrepreneur")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/physician/checkpoint-900/", adapter_name = "physician")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/psychologist/checkpoint-900/", adapter_name = "psychologist")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/salesman/checkpoint-2100/", adapter_name = "salesman")
lora_model.load_adapter(model_id = "/data1/dyf/agent_qwen_lora/scientist/checkpoint-1300/", adapter_name = "scientist")
tokenizer_text = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
model_text = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
model_text.eval()

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
        with lora_model.disable_adapter():
            generated_ids = lora_model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
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

        agent_string_list = ['historyExpert', 'advisor', 'storyWriter', 'engineer', 'entrepreneur', 'physician', 'psychologist', 'salesman', 'scientist']
        # instruction = "Generate representations for this word to be used to retrieve related articles: "
        #判断文本相似度确定使用哪个agent
        # sentences = [d2d_string, historyExpert1, historyExpert2, historyExpert3, advisor1, advisor2, advisor3, storyWriter1, storyWriter2, storyWriter3, engineer1,engineer2,engineer3, entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3]

        sentences = [expert_identity, historyExpert1, historyExpert2, historyExpert3, advisor1, advisor2, advisor3, storyWriter1, storyWriter2, storyWriter3, engineer1,engineer2,engineer3, entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3]
                # Load model from HuggingFace Hub
        # tokenizer_text = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        # model_text = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
        # model_text.eval()

        # Tokenize sentences
        encoded_input = tokenizer_text(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model_text(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        score_list = [sentence_embeddings[0] @ sentence_embeddings[i] for i in range(1, 28)]
        # print(score_list)
        #控制阈值，太低的话仍然使用基础模型
        # if max(score_list) <= 0.5:
        #     device = "cuda" # the device to load the model onto
        #     model_inputs = torch.load('tensor.pt')
        #     with lora_model.disable_adapter():
        #         generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens=5000, pad_token_id = 32000)
        #     tmp = pd.read_csv('agent_result.csv', index_col=0)
        #     tmp.loc['n', 'no_lora_times'] = tmp.loc['n', 'no_lora_times'] + 1
        #     tmp.to_csv('agent_result.csv')
        #均使用lora混合
        # 使用列表切片每隔三个数字取出一个子列表
        sub_lists = [score_list[i:i+3] for i in range(0, len(score_list), 3)]

        # 计算每个子列表的最大值
        max_values = [max(sub_list) for sub_list in sub_lists]

        # 获取前n个最大值及其对应的下标
        sorted_indices_values = sorted(enumerate(max_values), key=lambda x: x[1], reverse=True)[:n]
        adapter_index = []
        adapter_values = []
        adapter_string = []
        # 显示结果
        for index, value in sorted_indices_values:
            if float(value.numpy()) >= th: 
                adapter_index.append(index)
                adapter_values.append(float(value.numpy())) #保留一位小数
                adapter_string.append(agent_string_list[index])
        
        #softmax归一化
        if len(adapter_values) > 0:
            result = list(map(lambda x: x / T, adapter_values))
            adapter_values = result
            e_x = np.exp(adapter_values)
            summ = e_x.sum()
            normalized_data = e_x / summ
            normalized_data = normalized_data.tolist()
            # 保留一位小数
            normalized_data_rounded = [round(x, 1) for x in normalized_data]
        if len(adapter_string) > 1: #大于1 lora混合
            lora_model.base_model.add_weighted_adapter(adapters = adapter_string,weights = normalized_data_rounded,adapter_name = "merge",combination_type='cat')
            lora_model.set_adapter("merge")
            # logging.info(str(i) + 'pre: ' + agent_string)
            # logging.info('all')
            generated_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, do_sample = False) #seed = 1
        elif len(adapter_string) == 1: #等于1 找单独的lora
            lora_model.set_adapter(adapter_string[0])
            generated_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, do_sample = False)
        else:  #等于0 使用基础模型 
            with lora_model.disable_adapter():
                generated_ids = lora_model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
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
        # get spymaster hint word
        spymaster_prompt = task.get_input_prompt(i, method=method, role='spymaster')
        messages = [
            {"role":"system","content":''},
            {"role":"user","content":spymaster_prompt}
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to('cuda')
        # model.to(device)

        with lora_model.disable_adapter():
            generated_ids = lora_model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
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

        agent_string_list = ['historyExpert', 'advisor', 'storyWriter', 'engineer', 'entrepreneur', 'physician', 'psychologist', 'salesman', 'scientist']
        # instruction = "Generate representations for this word to be used to retrieve related articles: "
        #判断文本相似度确定使用哪个agent
        # sentences = [d2d_string, historyExpert1, historyExpert2, historyExpert3, advisor1, advisor2, advisor3, storyWriter1, storyWriter2, storyWriter3, engineer1,engineer2,engineer3, entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3]

        sentences = [expert_identity, historyExpert1, historyExpert2, historyExpert3, advisor1, advisor2, advisor3, storyWriter1, storyWriter2, storyWriter3, engineer1,engineer2,engineer3, entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3]
                # Load model from HuggingFace Hub
        # tokenizer_text = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        # model_text = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
        # model_text.eval()

        # Tokenize sentences
        encoded_input = tokenizer_text(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model_text(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        score_list = [sentence_embeddings[0] @ sentence_embeddings[i] for i in range(1, 28)]
        # print(score_list)
        #控制阈值，太低的话仍然使用基础模型
        # if max(score_list) <= 0.5:
        #     device = "cuda" # the device to load the model onto
        #     model_inputs = torch.load('tensor.pt')
        #     with lora_model.disable_adapter():
        #         generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens=5000, pad_token_id = 32000)
        #     tmp = pd.read_csv('agent_result.csv', index_col=0)
        #     tmp.loc['n', 'no_lora_times'] = tmp.loc['n', 'no_lora_times'] + 1
        #     tmp.to_csv('agent_result.csv')
        #均使用lora混合
        # 使用列表切片每隔三个数字取出一个子列表
        sub_lists = [score_list[i:i+3] for i in range(0, len(score_list), 3)]

        # 计算每个子列表的最大值
        max_values = [max(sub_list) for sub_list in sub_lists]

        # 获取前n个最大值及其对应的下标
        sorted_indices_values = sorted(enumerate(max_values), key=lambda x: x[1], reverse=True)[:n]
        adapter_index = []
        adapter_values = []
        adapter_string = []
        # 显示结果
        for index, value in sorted_indices_values:
            if float(value.numpy()) >= th: 
                adapter_index.append(index)
                adapter_values.append(float(value.numpy())) #保留一位小数
                adapter_string.append(agent_string_list[index])
        
        #softmax归一化
        if len(adapter_values) > 0:
            result = list(map(lambda x: x / T, adapter_values))
            adapter_values = result
            e_x = np.exp(adapter_values)
            summ = e_x.sum()
            normalized_data = e_x / summ
            normalized_data = normalized_data.tolist()
            # 保留一位小数
            normalized_data_rounded = [round(x, 1) for x in normalized_data]
        if len(adapter_string) > 1: #大于1 lora混合
            lora_model.base_model.add_weighted_adapter(adapters = adapter_string,weights = normalized_data_rounded,adapter_name = "merge",combination_type='cat')
            lora_model.set_adapter("merge")
            # logging.info(str(i) + 'pre: ' + agent_string)
            # logging.info('all')
            generated_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, do_sample = False) #seed = 1
        elif len(adapter_string) == 1: #等于1 找单独的lora
            lora_model.set_adapter(adapter_string[0])
            generated_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, do_sample = False)
        else:  #等于0 使用基础模型 
            with lora_model.disable_adapter():
                generated_ids = lora_model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
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

        with lora_model.disable_adapter():
            generated_ids = lora_model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
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

        agent_string_list = ['historyExpert', 'advisor', 'storyWriter', 'engineer', 'entrepreneur', 'physician', 'psychologist', 'salesman', 'scientist']
        # instruction = "Generate representations for this word to be used to retrieve related articles: "
        #判断文本相似度确定使用哪个agent
        # sentences = [d2d_string, historyExpert1, historyExpert2, historyExpert3, advisor1, advisor2, advisor3, storyWriter1, storyWriter2, storyWriter3, engineer1,engineer2,engineer3, entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3]

        sentences = [expert_identity, historyExpert1, historyExpert2, historyExpert3, advisor1, advisor2, advisor3, storyWriter1, storyWriter2, storyWriter3, engineer1,engineer2,engineer3, entrepreneur1,entrepreneur2,entrepreneur3,physician1,physician2,physician3,psychologist1,psychologist2,psychologist3,salesman1,salesman2,salesman3,scientist1,scientist2,scientist3]
                # Load model from HuggingFace Hub
        # tokenizer_text = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        # model_text = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
        # model_text.eval()

        # Tokenize sentences
        encoded_input = tokenizer_text(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model_text(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        score_list = [sentence_embeddings[0] @ sentence_embeddings[i] for i in range(1, 28)]
        # print(score_list)
        #控制阈值，太低的话仍然使用基础模型
        # if max(score_list) <= 0.5:
        #     device = "cuda" # the device to load the model onto
        #     model_inputs = torch.load('tensor.pt')
        #     with lora_model.disable_adapter():
        #         generated_ids, agent_string, agent_ids = lora_model.generate(model_inputs, max_new_tokens=5000, pad_token_id = 32000)
        #     tmp = pd.read_csv('agent_result.csv', index_col=0)
        #     tmp.loc['n', 'no_lora_times'] = tmp.loc['n', 'no_lora_times'] + 1
        #     tmp.to_csv('agent_result.csv')
        #均使用lora混合
        # 使用列表切片每隔三个数字取出一个子列表
        sub_lists = [score_list[i:i+3] for i in range(0, len(score_list), 3)]

        # 计算每个子列表的最大值
        max_values = [max(sub_list) for sub_list in sub_lists]

        # 获取前n个最大值及其对应的下标
        sorted_indices_values = sorted(enumerate(max_values), key=lambda x: x[1], reverse=True)[:n]
        adapter_index = []
        adapter_values = []
        adapter_string = []
        # 显示结果
        for index, value in sorted_indices_values:
            if float(value.numpy()) >= th: 
                adapter_index.append(index)
                adapter_values.append(float(value.numpy())) #保留一位小数
                adapter_string.append(agent_string_list[index])
        
        #softmax归一化
        if len(adapter_values) > 0:
            result = list(map(lambda x: x / T, adapter_values))
            adapter_values = result
            e_x = np.exp(adapter_values)
            summ = e_x.sum()
            normalized_data = e_x / summ
            normalized_data = normalized_data.tolist()
            # 保留一位小数
            normalized_data_rounded = [round(x, 1) for x in normalized_data]
        if len(adapter_string) > 1: #大于1 lora混合
            lora_model.base_model.add_weighted_adapter(adapters = adapter_string,weights = normalized_data_rounded,adapter_name = "merge",combination_type='cat')
            lora_model.set_adapter("merge")
            # logging.info(str(i) + 'pre: ' + agent_string)
            # logging.info('all')
            generated_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, do_sample = False) #seed = 1
        elif len(adapter_string) == 1: #等于1 找单独的lora
            lora_model.set_adapter(adapter_string[0])
            generated_ids = lora_model.generate(model_inputs, max_new_tokens = 5000, do_sample = False)
        else:  #等于0 使用基础模型 
            with lora_model.disable_adapter():
                generated_ids = lora_model.generate(model_inputs, max_new_tokens=5000, do_sample = False)
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