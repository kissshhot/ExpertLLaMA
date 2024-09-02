import json
import jsonlines
import os
file='trivia_creative_writing_100_n_5.jsonl'
DATA_PATH = './agent/SPP/data'
new_data = []
path = os.path.join(DATA_PATH, 'trivia_creative_writing', file)
with open(path, "r") as f:
    data = [json.loads(line) for line in f]
for i in range(0, 100):
    for j in range(0, 5):
        new_dic = {'questions': [data[i]['questions'][j]], 'answers': [data[i]['answers'][j]], 'question_ids': [data[i]['question_ids'][j]], 'topic': data[i]['topic']}
        new_data.append(new_dic)
path = os.path.join(DATA_PATH, 'trivia_creative_writing', 'trivia_creative_writing_100_n_1_copy.jsonl')
for line in new_data:
    with jsonlines.open(path, mode='a') as writer:
        writer.write(line)