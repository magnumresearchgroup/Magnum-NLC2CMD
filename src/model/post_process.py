import json
from submission_code.nlp_tools import tokenizer

MAPPING = {
    "_FILE": "File",
    "_DIRECTORY": "Directory",
    "_PATH": "Path",
    "_PERMISSION": "Permission",
    "_USERNAME": "Username",
    "_GROUPNAME": "Groupname",
    "_DATETIME": "DateTime",
    "_NUMBER": "Number",
    "_SIZE": "Size",
    "_TIMESPAN": "Timespan",
    "_REGEX": "Regex"
}


def post_process():
    with open('src/data/test_data.json') as f:
        test_data = json.load(f)

    with open('pred_2000.txt') as f:
        bash_preds = f.read().split('\n')

    english_text = []
    for i in test_data:
        english_text.append(test_data[i]["invocation"])

    replaced_bash = []
    for text, cmd in zip(english_text, bash_preds):
        rep_cmd = cmd
        for argument, filler in list(tokenizer.ner_tokenizer(text)[1][0].values()):
            rep_cmd = rep_cmd.replace(MAPPING[filler], argument, 1)
        replaced_bash.append(rep_cmd)

    with open('replaced_pred_2000.txt', 'w') as f:
        f.write("\n".join(replaced_bash))