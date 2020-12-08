from nl2bash.bashlint.data_tools import bash_tokenizer
from nl2bash.nlp_tools2 import tokenizer


def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]


def tokenize_bash(text):
    return bash_tokenizer(text)


english_txt = open("/tmp/pycharm_project_78/submission_code/src/submission_code/nl2cmd/invocations.txt",
                   encoding="utf8").read().split("\n")
bash_txt = open("/tmp/pycharm_project_78/submission_code/src/submission_code/nl2cmd/cmds.txt",
                encoding="utf8").read().split("\n")

processed_cmd = []
processed_nl = []

for cmd in bash_txt:
    temp_cmd = tokenize_bash(cmd)
    new_cmd = ' '.join(temp_cmd)
    processed_cmd.append(new_cmd)

for nl in english_txt:
    temp_nl = tokenize_eng(nl)
    new_nl = ' '.join(temp_nl)
    processed_nl.append(new_nl)

outF = open("/tmp/pycharm_project_78/submission_code/src/submission_code/nl2cmd/cmds_proccess.txt", "w")
for line in processed_cmd:
    outF.write(line)
    outF.write("\n")
outF.close()

outF = open("/tmp/pycharm_project_78/submission_code/src/submission_code/nl2cmd/invocations_proccess.txt", "w")
for line in processed_nl:
    outF.write(line)
    outF.write("\n")
outF.close()

print("done")
