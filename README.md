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
2. git clone --recursive https://github.com/magnumresearchgroup/Magnum-NLC2CMD.git
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
2. You can change the `gpu=-1` in `main.py` to `gpu=0`, and replace the code in `main.py` accordingly with the following code for faster inference time
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

## References

* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [Bashlex](https://github.com/idank/bashlex)
* [Clai](https://github.com/IBM/clai)
* [Tellina](https://github.com/TellinaTool/nl2bash)
* [Training Tips for the Transformer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf)

## License

See the [LICENSE](https://github.com/QuchenFu/Magnum-NLC2CMD/blob/final/LICENSE) file for license rights and limitations (MIT).