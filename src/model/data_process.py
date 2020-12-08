import json
import random
import os
from bashlint.data_tools import bash_tokenizer
from submission_code.nlp_tools import tokenizer


def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]

def tokenize_bash(text):
    return bash_tokenizer(text,  loose_constraints=True, arg_type_only=False)


def preprocess(data_dir, data_file):
    data = {}
    with open(os.path.join(data_dir,data_file)) as f:
        raw_data = json.load(f)
    for i in  range(1, len(raw_data.keys())+1):
        data[str(i)] = raw_data[str(i)]
        data[str(i)]['cmd'] = [raw_data[str(i)]['cmd']]

    rand_seed = 94726
    random.seed(rand_seed)
    train_data, test_data = {}, {}
    all_index = [i for i in range(1, len(data.keys())+1)]
    random.shuffle(all_index)
    for i in all_index[:int(len(all_index)*0.8)]:
        train_data[str(i)] = data[str(i)]
    for j in all_index[int(len(all_index)*0.8):]:
        test_data[str(j)] = data[str(j)]

    with open('src/data/train_data.json', 'w') as f:
        json.dump(train_data, f)
    with open('src/data/test_data.json', 'w') as f:
        json.dump(test_data, f)

    for split, data in zip(['train', 'test'],[train_data, test_data]):
        english_txt = []
        bash_txt = []
        for i in data:
            english_txt.append(data[i]['invocation'])
            bash_txt.append(data[i]['cmd'][0])

        processed_cmd = []
        processed_nl = []

        for cmd, nl in zip(bash_txt, english_txt):
            processed_cmd.append(' '.join(tokenize_bash(cmd)))
            processed_nl.append(' '.join(tokenize_eng(nl)))

        with open('{}/cmds_proccess_{}.txt'.format(data_dir, split), 'w') as outF:
            for line in processed_cmd:
                outF.write(line)
                outF.write("\n")

        with open('{}/invocations_proccess_{}.txt'.format(data_dir, split), 'w') as outF:
            for line in processed_nl:
                outF.write(line)
                outF.write("\n")