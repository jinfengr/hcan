## Bridging the Gap Between Relevance Matching and Semantic Matching for Short Text Similarity Modeling
This repo contains code and data for our paper published in [EMNLP'19](https://jinfengr.github.io/publications/Rao_etal_EMNLP2019.pdf).

### Reference
If you are using this code or dataset, please kindly cite the paper below:
```
@article{rao2019hcan,
  title={Bridging the Gap Between Relevance Matching and Semantic Matching for Short Text Similarity Modeling},
  author={Rao, Jinfeng and Liu, Linqing and Tay, Yi and Yang, Wei and Shi, Peng and Lin, Jimmy}
  journal={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```


### Requirements
- Python 2.7
- Tensorflow (tested on 1.9.0)
- Keras (tested on 2.1.5)

### Install
- Download our repo:
```
git clone https://github.com/jinfengr/hcan.git
cd hcan
```
- Install Tensorflow and Keras dependency:
```
$ pip install -r requirements.txt
```
- Install [gdrive](https://github.com/prasmussen/gdrive)
- Download required data and word2vec:
```
$ chmod +x *.sh; ./download.sh
$ ./generate_idf.sh
```

### Run
- Run on TrecQA/Quora/TwitterURL datasets:
```
CUDA_VISIBLE_DEVICES=0 python -u train.py --dataset TrecQA -j hcan
```
The path of best model and output predictions will be shown in the log. 

- Run on Twitter datasets (test on `trec-2013`):
```
CUDA_VISIBLE_DEVICES=0 python -u train.py --dataset twitter -t trec-2013 -j hcan
```
Note: you might need around ~40GB memory to create the twitter dataset (because of the large size of IDF weights). Please file a issue if you have any problem in creating the dataset.

- Parameter sweep to find the best parameter set (make sure the dataset is created before sweep):
```
./param_sweep.sh TrecQA hcan 0 &
```
This command will save all the outputs under tune_logs folder. 


### Command line parameters
| option                   | input format |   default   | description |
|--------------------------|--------------|-------------|-------------|
| `-l`   | [true, false]       | false     | whether to load pre-created dataset (set to true when data is ready) |
| `-j` | [matching, biattention, hcan]       | matching     | attention choices, matching for relevance matching in Sec. 2.2, biattention for semantic matching in Sec. 2.3, hcan for the complete hcan model |
| `-e` | [deepconv, wideconv, bilstm]       | deepconv     | encoder choices described in Sec. 2.1 |
| `-w` | [none, query]       | none     | whether to include IDF weighting, none for not include, query for include |
| `--nb_layers`    | [1, n)    | 5 | number of convolutional or BiLSTM layers |
| `--nb_filters`    | [1, n)    | 256 | number of convolutional filters or BiLSTM hidden dim |
| `--model_option`| [complete, word-only]       | complete | what input sources to use, complete for using both word and character-level ngram representations, word-only for using only word representations  |
| `--conv_option` | [normal, ResNet]       | normal     | convolutional model, normal or ResNet |
| `--co-attention`    | [BiDAF, ESIM]   | BiDAF | different biattention implementations |
| `--highway` | [true, false] | false | whether to include highway layer |
| `-t`   | [trec-2011, trec-2012, trec-2013, trec-2014] | trec-2013 | test set, only needed for twitter datasets|
| `--load_model`     | [true, false]       | false     | whether to load pre-trained model |
| `-b`   | [1, n)    | 64 | batch size | 
| `-d`    | [0, 1]    | 0.1 | dropout rate | 
| `-o`    | [sgd, adam, rmsprop] | sgd | optimization method | 
| `--lr`  | [0, 1]    | 0.05 | learning rate |
| `--epochs`| [1, n)  | 15   | number of training epochs | 
| `--trainable` | [true, false] | true | whether to train word embeddings | 
| `--val_split` | (0, 1) | 0.15 | percentage of validation set sampled from training set | 
| `-v`| [0, 1, 2] | 1 | verbose (for logging), 0 for silent, 1 for interactive, 2 for per-epoch logging |
