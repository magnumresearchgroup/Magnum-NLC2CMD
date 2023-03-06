<!---
# <img src="https://www.magnum.io/img/magnum.png" width="24" height="24"> Magnum-NLC2CMD 

<img src="https://evalai.s3.amazonaws.com/media/logos/4c055dbb-a30a-4aa1-b86b-33dd76940e14.jpg" align="right"
     alt="Magnum logo" height="150">

Magnum-NLC2CMD is the winning solution for the **[NeurIPS 2020 NLC2CMD challenge]**. The solution was produced by Quchen Fu and Zhongwei Teng, researchers in the **[Magnum Research Group]** at Vanderbilt University. The Magnum Research Group is part of the **[Institute for Software Integrated Systems]**. 

The NLC2CMD Competition challenges you to build an algorithm that can translate an English description (𝑛𝑙𝑐) of a command line task to its corresponding command line syntax (𝑐). The model achieved a 0.53 score in Accuracy Track on the open **[Leaderboard]**. The  **[Tellina]** model was the previous SOTA which was used as the baseline.
<p align="left">
<img width="650" alt="Screen Shot 2020-11-23 at 3 38 13 PM" src="https://user-images.githubusercontent.com/31392274/100018358-f34fa600-2da1-11eb-94c6-b848c774aca9.png">
</p>

[NeurIPS 2020 NLC2CMD challenge]: http://nlc2cmd.us-east.mybluemix.net/#/
[Magnum Research Group]:https://www.magnum.io
[Institute for Software Integrated Systems]:https://www.isis.vanderbilt.edu
[leaderboard]: https://eval.ai/web/challenges/challenge-page/674/leaderboard/1831
[tellina]: https://github.com/IBM/clai/tree/master/clai/server/plugins/tellina
--->
[chatGPT achieved an accuracy score of 80.6% on the test set under zeroshot conditions]: https://arxiv.org/abs/2302.07845

## Rethinking NL2CMD in the age of ChatGPT 

There is a widespread belief among experts that the field of natural language processing (NLP) is currently experiencing a paradigm shift as a result of the introduction of LLM (Large Language Models), with chatGPT being the leading example of this new technology. With this new technology, many tasks that previously relied on fine-tuning pre-trained models can now be achieved through prompt engineering, which involves identifying the appropriate instructions to direct the language model (LLM) for specific tasks. To evaluate the effectiveness of chatGPT, we conducted tests on the original NL2BASH dataset, and the results were exceptional. Specifically, we found that **[chatGPT achieved an accuracy score of 80.6% on the test set under zeroshot conditions]**. Although there are concerns about the possibility of data leakage in LLM-based translation due to the vast amount of internet text in the pre-training data, we have confidence in the performance of chatGPT, given its consistent ability to achieve scores of 80% or higher across all training, testing, and evaluation datasets. 

<p align="center">
<img width="500" alt="pipeline" src="https://user-images.githubusercontent.com/31392274/223152672-4704ed94-83d1-4ff2-93d2-14dab48ab748.png">
</p>

We have conducted further exploration into the potential of streamlining our data generation pipeline with the assistance of ChatGPT, as shown in Figure. In order to generate Bash commands, we utilized the prompt Generate bash command and do not include example. We set the ”temperature” parameter to 1 for maximum variability. These generated commands were then subjected to a de-duplication script, resulting in a surprisingly low duplicate rate of 6% despite prompting the system 44671 times. Subsequently, the data were validated using the same bash parsing tool previously mentioned, and 41.7% of the generated bash commands were deemed valid. The preprocessed bash commands were combined with the prompt Translate to English, yielding a paired English-Bash dataset with a size of 17050. We set the temperature parameter to 0 for reproduciblity. 

In order to assess the quality of this generated dataset, we tested the performance of augmenting the original dataset with the generated version NL2CMD: An Updated Workflow for Natural Language to Bash Commands Translation 31 and found no performance drop. We further tested this approach by setting the temperature parameter to 1 to introduce more variability, which yielded different English sentences for each Bash command, serving as a useful data augmentation tool. 

This suggests that the ChatGPT-generated dataset is of higher quality than our previous pipeline. Furthermore, the performance of training on generated data and evaluating on NL2Bash was greatly improved, with the score increasing from -13% to approximately 10%. It is important to note that this is only a preliminary exploration into using ChatGPT as a data generation tool, and our observations represent a lower-bound on the potential benefits of this method. 

What is particularly groundbreaking about this approach is the efficiency with which it was implemented. Whereas the previous pipeline took two months to build, the ChatGPT streamlined version was completed in just three days. We have made our code and dataset available on Github. Notably, the distribution of generated utilities displayed a much smaller long tail effect, suggesting that it more accurately captures the command usage distribution.

## Requirements
<details><summary>Show details</summary>
<p>

* numpy
* six
* nltk
* experiment-impact-tracker
* scikit-learn
* pandas
* flake8==3.8.3
* spacy==2.3.0
* tb-nightly==2.3.0a20200621
* tensorboard-plugin-wit==1.6.0.post3
* torch==1.6.0
* torchtext==0.4.0
* torchvision==0.7.0
* tqdm==4.46.1
* OpenNMT-py==2.0.0rc2

</p>
</details>

## How it works

### Environment
1. Create a virtual environment with python3.6 installed(`virtualenv`)
2. `git clone --recursive https://github.com/magnumresearchgroup/Magnum-NLC2CMD.git`
3. use `pip3 install -r requirements.txt` to install the two requirements files.


### Data pre-processing
1. Run `python3 main.py --mode preprocess --data_dir src/data --data_file nl2bash-data.json` and `cd src/model && onmt_build_vocab -config nl2cmd.yaml -n_sample 10347 --src_vocab_threshold 2 --tgt_vocab_threshold 2` to process raw data.
2. You can also download the Original raw data [here](https://ibm.ent.box.com/v/nl2bash-data)


### Train
1. ``cd src/model && onmt_train -config nl2cmd.yaml``
2. Modify the `world_size` in `src/model/nl2cmd.yaml` to the number of GPUs you are using and put the ids as `gpu_ranks`.
4. You can also download one of our pre-trained model [here](https://drive.google.com/file/d/1HXg2j1QuuDBV-8vpj2YdBhBK81pLK7bg/view?usp=sharing)

### Inference
2. `onmt_translate -model src/model/run/model_step_2000.pt -src src/data/invocations_proccess_test.txt -output pred_2000.txt -gpu 0 -verbose`

### Evaluate
1. `python3 main.py --mode eval --annotation_filepath src/data/test_data.json --params_filepath src/configs/core/evaluation_params.json --output_folderpath src/logs --model_dir src/model/run  --model_file model_step_2400.pt model_step_2500.pt`
2. You can change the `gpu=-1` in `src/model/predict.py` to `gpu=0`, and replace the code in `src/model/predict.py` accordingly with the following code for faster inference time
    <details><summary>Show details</summary>
    <p>
    
    ```
    invocations = [' '.join(tokenize_eng(i)) for i in invocations]
    translated = translator.translate(invocations, batch_size=n_batch)
    commands = [t[:result_cnt] for t in translated[1]]
    confidences = [ np.exp( list(map(lambda x:x.item(), t[:result_cnt])) )/2 for t in translated[0]]
    for i in range(len(confidences)):
        confidences[i][0] = 1.0
    ```
    </p>
    </details>
    
## Metrics

### Accuracy metric

𝑆𝑐𝑜𝑟𝑒(𝐴(𝑛𝑙𝑐))=max𝑝∈𝐴(𝑛𝑙𝑐)𝑆(𝑝) if ∃𝑝∈𝐴(𝑛𝑙𝑐) such that 𝑆(𝑝)>0;
 
𝑆𝑐𝑜𝑟𝑒(𝐴(𝑛𝑙𝑐))=1|𝐴(𝑛𝑙𝑐)|∑𝑝∈𝐴(𝑛𝑙𝑐)𝑆(𝑝) otherwise.

### Reproduce

1. We used 2x `Nvidia 2080Ti GPU` + 64G memory machine running `Ubuntu 18.04 LTS`
2. Change the `batch_size` in `nl2cmd.yaml` to the largest your GPU can support without `OOM error`
2. Train multiple models by modify `seed` in `nl2cmd.yaml`, you should also modify the `save_model` to avoid overwrite existing models.
3. Hand pick the best performed ones on local test set and put their directories in the main.py

### New dataset

https://github.com/magnumresearchgroup/bash_gen

## References

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [Bashlex](https://github.com/idank/bashlex)
* [Clai](https://github.com/IBM/clai)
* [Tellina](https://github.com/TellinaTool/nl2bash)
* [Training Tips for the Transformer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf)

## Acknowledgment

This work was supported in part by NSF Award# 1552836, At-scale analysis of issues in cyber-security and software engineering.

## License

See the [LICENSE](https://github.com/QuchenFu/Magnum-NLC2CMD/blob/final/LICENSE) file for license rights and limitations (MIT).

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=magnumresearchgroup/Magnum-NLC2CMD&type=Date)](https://star-history.com/#magnumresearchgroup/Magnum-NLC2CMD&Date)

## Reference
If you use this repository, please consider citing:

```
@article{Fu2021ATransform,
  title={A Transformer-based Approach for Translating Natural Language to Bash Commands},
  author={Quchen Fu and Zhongwei Teng and Jules White and Douglas C. Schmidt},
  journal={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)},
  year={2021},
  pages={1241-1244}
}
```
```
@article{fu2023nl2cmd,
  title={NL2CMD: An Updated Workflow for Natural Language to Bash Commands Translation},
  author={Fu, Quchen and Teng, Zhongwei and Georgaklis, Marco and White, Jules and Schmidt, Douglas C},
  journal={Journal of Machine Learning Theory, Applications and Practice},
  pages={45--82},
  year={2023}
}
