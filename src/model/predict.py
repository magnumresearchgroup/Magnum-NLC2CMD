from submission_code.nlp_tools import tokenizer
from onmt.translate.translator import build_translator
from argparse import Namespace
import math
import os




def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]


def predict(invocations, model_dir, model_file, result_cnt=5):
    """
    Function called by the evaluation script to interface the participants submission_code
    `predict` function accepts the natural language invocations as input, and returns
    the predicted commands along with confidences as output. For each invocation,
    `result_cnt` number of predicted commands are expected to be returned.

    Args:
        1. invocations : `list (str)` : list of `n_batch` (default 16) natural language invocations
        2. result_cnt : `int` : number of predicted commands to return for each invocation

    Returns:
        1. commands : `list [ list (str) ]` : a list of list of strings of shape (n_batch, result_cnt)
        2. confidences: `list[ list (float) ]` : confidences corresponding to the predicted commands
                                                 confidence values should be between 0.0 and 1.0.
                                                 Shape: (n_batch, result_cnt)
    """
    opt = Namespace(models=[
        os.path.join(model_dir, file) for file in model_file
    ], n_best=5,
        avg_raw_probs=False,
        alpha=0.0, batch_type='sents', beam_size=5,
        beta=-0.0, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', fp32=True,
        gpu=-1, ignore_when_blocking=[], length_penalty='none', max_length=100, max_sent_length=None,
        min_length=0, output='/dev/null', phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1,
        ratio=-0.0, replace_unk=True, report_align=False, report_time=False, seed=829, stepwise_penalty=False,
        tgt=None, verbose=False, tgt_prefix=None)
    translator = build_translator(opt, report_score=False)


    n_batch = len(invocations)
    commands = [
        [''] * result_cnt
        for _ in range(n_batch)
    ]
    confidences = [
        [1, 0, 0, 0, 0]
        for _ in range(n_batch)
    ]

    ################################################################################################
    #     Participants should add their codes to fill predict `commands` and `confidences` here    #
    ################################################################################################
    for idx, inv in enumerate(invocations):
        new_inv = tokenize_eng(inv)
        new_inv = ' '.join(new_inv)
        translated = translator.translate([new_inv], batch_size=1)
        for i in range(result_cnt):
            commands[idx][i] = translated[1][0][i]
            confidences[idx][i] = math.exp(translated[0][0][i].item()) / 2
        confidences[idx][0] = 1.0
    ################################################################################################
    #                               Participant code block ends                                    #
    ################################################################################################
    return commands, confidences
