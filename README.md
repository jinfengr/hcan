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
- Tensorflow or Theano (tested on TF 1.4.1)
- Keras (tested on 2.1.5)

### Install
- Download our repo:
```
git clone https://github.com/Jeffyrao/hcan.git
cd hcan
```
- Install [gdrive](https://github.com/prasmussen/gdrive)
- Download required data and word2vec:
```
$ chmod +x download.sh; ./download.sh
```
- Install Tensorflow and Keras dependency:
```
$ pip install -r requirements.txt
```

### Run
- Run on TrecQA/Quora/TwitterURL datasets:
```
CUDA_VISIBLE_DEVICES=0 python -u train.py -d TrecQA
```
The path of best model and output predictions will be shown in the log. Default parameters should work reasonably well.

- Run on Twitter datasets:
```
CUDA_VISIBLE_DEVICES=0 python -u train.py -d TrecQA -t trec-2013
```
Note: you might need around ~40GB memory to create the twitter dataset (because of the large size of IDF weights). Please file a issue if you have any problem in creating the dataset.

- Parameter sweep to find the best parameter set:
```
./param_sweep.sh TrecQA join 0 &
```
This command will save all the outputs under tune-logs folder. 

### Evaluate with trec_eval
Example with twitter pred_file:
```
$ ./trec_eval.8.1/trec_eval data/twitter/qrels.microblog2011-2014.txt \
                            {pred_file}
```

### Command line parameters
| option                   | input format |   default   | description |
|--------------------------|--------------|-------------|-------------|
| `-t`   | [trec-2011, trec-2012, trec-2013, trec-2014] | trec-2013 | test set, only needed for twitter datasets|
| `-l`   | [true, false]       | false     | whether to load pre-created dataset (set to true when data is ready) |
| `--load_model`     | [true, false]       | false     | whether to load pre-trained model |
| `-b`   | [1, n)    | 64 | batch size | 
| `-n`    | [1, n)    | 256 | number of convolutional filters |
| `-d`    | [0, 1]    | 0.1 | dropout rate | 
| `-o`    | [sgd, adam, rmsprop] | sgd | optimization method | 
| `--lr`  | [0, 1]    | 0.05 | learning rate |
| `--epochs`| [1, n)  | 15   | number of training epochs | 
| `--trainable` | [true, false] | true | whether to train word embeddings | 
| `--val_split` | (0, 1) | 0.15 | percentage of validation set sampled from training set | 
| `-v`| [0, 1, 2] | 1 | verbose (for logging), 0 for silent, 1 for interactive, 2 for per-epoch logging |
| `--conv_option` | [normal, ResNet]       | normal     | convolutional model, normal or ResNet |
| `--model_option`| [complete, word-url]       | complete | what input sources to use, complete for MP-HCNN, word-url for only modeling query-tweet (word) and query-url (char)  |

