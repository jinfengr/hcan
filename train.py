import random
random.seed(123456789)
import numpy as np
np.random.seed(123456789)
import subprocess
import shlex
import sys
import math
from optparse import OptionParser
from collections import defaultdict
import pprint
import pdb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K, optimizers

# Fix random seed for reproducible experiments
if K.backend() == "tensorflow":
    import tensorflow as tf
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) #, device_count={"GPU": 0}
    tf.set_random_seed(1234)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
else:
    from theano.tensor.shared_randomstreams import RandomStreams
    # from theano import function
    srng = RandomStreams(seed=123456789)
    srng.seed(123456789)  # seeds rv_u and rv_n with different seeds each


from utils import invert_dict, unsplit_query, merge_two_dicts, sample_aaai_val_set
from data_preprocess import gen_data, load_data, save_data, construct_vocab_emb
from attention_model import create_attention_model


def evaluate(predictions_file, qrels_file):
    pargs = shlex.split("/bin/sh run_eval.sh '{}' '{}'".format(predictions_file, qrels_file))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()

    print(perr)
    if sys.version_info[0] < 3:
        lines = pout.split(b'\n')
    else:
        lines = pout.split(b'\n')
    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    p30 = float(lines[2].strip().split()[-1])
    return map, mrr, p30


def config_dataset(args):
    # dataset-specific configuration
    dataset_name = args['dataset']
    val_name, val_set, with_url = None, None, False
    if dataset_name == 'twitter':
        args['raw_data'] = "../data/twitter"
        train_name, test_name = args['train'], args['test']
        if train_name == 'train_all':
            train_set = ['trec-2011', 'trec-2012', 'trec-2013', 'trec-2014']
            train_set.remove(test_name)
        else:
            train_set = [train_name]
        test_set = [test_name]
        num_classes = 2
        with_url = True
        print(test_set)
    elif dataset_name == 'TrecQA':
        train_name, val_name, test_name = 'train-all', 'raw-dev', 'raw-test'
        # if args['test'] == 'clean-test':
        #    train_name, val_name, test_name = 'train-all', 'clean-dev', 'clean-test'
        train_set, val_set, test_set = [train_name], [val_name], [test_name]
        num_classes = 2
    elif dataset_name == 'Quora':
        train_name, val_name, test_name = 'train', 'dev', 'test'
        train_set, val_set, test_set = [train_name], [val_name], [test_name]
        num_classes = 2
    elif dataset_name == 'TwitterURL':
        train_name, test_name = 'train', 'test'
        train_set, test_set = [train_name], [test_name]
        num_classes = 2
    else:
        print("Error dataset!")
        sys.exit(0)
    return train_name, val_name, test_name, train_set, val_set, test_set, num_classes, with_url


def print_dataset(mode, dataset, vocab_inv):
    for key in dataset:
        if dataset[key].size == 0: continue
        print(key, dataset[key].shape)
        if "weight" in key:
            print(key, dataset[key][0][:2])
        #elif "mask" in key:
        #    print(key, dataset[key][0])
        elif "word" in key:
            print(dataset[key][0])
            print(key, unsplit_query(dataset[key][0], "word", vocab_inv["word"]))
        elif "url" in key:
            print(key, unsplit_query(dataset[key][0], "3gram", vocab_inv["3gram"]))
        elif "3gram" in key:
            print(key, unsplit_query(dataset[key][0], "3gram", vocab_inv["3gram"]))
        elif "feat" in key:
            print(key, dataset[key][0])


def get_model_weights(model):
    model_weights = 0.0
    for layer in model.layers:
        for weights in layer.get_weights():
            model_weights += np.sum(weights)
    return model_weights


def batch_generator(dataset, batch_size):
    while True:
        num_batches = int(math.ceil(dataset['sim'].shape[0] * 1.0 / batch_size))
        for i in range(num_batches):
            start_idx, end_idx = i*batch_size, min((i+1)*batch_size, dataset['sim'].shape[0])
            x, y = {}, dataset['sim'][start_idx:end_idx]
            for key in dataset:
                x[key] = dataset[key][start_idx:end_idx]
            yield (x, y)


def get_default_args():
    return {"load_data": False, # load previously created data to save the time for data conversion
            "load_model": False, # load previously trained model for testing
            "mode": 'hcan', # model name
            "dataset": 'TrecQA', # dataset name, TrecQA, Quora, TwitterURL, twitter
            "train": "train_all", # train set folder
            "test": "raw-test", # test set folder
            "trainable": True, # whether to train the word embedding or not
            "nb_filters": 128, # number of filters in the CNN model
            "nb_layers": 5, # number of layers in the CNN model
            "dropout": 0.3,
            "batch_size": 256,
            'weighting': 'none', # "query" or "doc" or "none":
            "embedding": 'word2vec',
            # "query" -> only weight query words and ngrams,
            # "doc" -> weight both query and document words/ngrams
            # whether to mask padded word or not -- this param seems have a big impact on model performance
            "mask": False,
            "join": "matching", # matching for MP-HCNN, biattention for only co_attention, hcan for both
            "co_attention": 'BiDAF', # can be BiDAF, ESIM, DecompAtt
            "val_split": 0.15,
            # "word_only, ""word_url" -> query-doc (word) + query-url; "complete" -> "query-doc (word+ngram) + query-url"
            "model_option": "complete",
            "encoder_option": "deepconv",
            "highway": False,
            # "normal" -> conv connection layer by layer; can also be "ResNet"; "DenseNet (CVPR 2017 best paper)";
            "conv_option": "normal",
            # change the base path to your path correspondingly
            "experimental_data": "./experimental-data",
            "raw_data": "../data/twitter/",
            "qrels_file": "../data/twitter/qrels.all.txt",
            "base_embed_path": "../data/word2vec/GoogleNews-vectors-negative300.bin.gz",
            # "base_embed_path": "../data/word2vec/tweet_vector_0401.bin",
            "external_feat": False,
            "norm_weight": False,
            "cos": False,
            "epochs": 15,
            "optimizer": "sgd",
            "learning_rate": 0.05,
            "verbose": 1}


def print_args(args):
    print('------------------------------------------------------------')
    pprint.pprint(args)
    print('------------------------------------------------------------')


def set_args(args, options):
    print(type(options))
    for arg in dir(options):
        if arg in args and getattr(options, arg) is not None:
            args[arg] = getattr(options, arg)
    if args['embedding'] == 'glove':
        args['base_embed_path'] = "../data/word2vec/gensim.glove.840B.300d.txt"


def create_option_parser():
    parser = OptionParser()
    parser.add_option("-n", "--nb_filters", action="store", type=int, dest="nb_filters")
    parser.add_option("--nb_layers", "--nb_layers", action="store", type=int, dest="nb_layers")
    parser.add_option("-d", "--dropout", action="store", type=float, dest="dropout")
    parser.add_option("-b", "--batch_size", action="store", type=int, dest="batch_size")
    parser.add_option("-w", "--weighting", action="store", type=str, dest="weighting")
    parser.add_option("-m", "--mask", action="store_true", dest="mask")
    parser.add_option("-j", "--join", action="store", type=str, dest="join")
    parser.add_option("-t", "--test", action="store", type="string", dest="test")
    parser.add_option("-c", "--co_attention", action="store", type="string", dest="co_attention")
    parser.add_option("--emb", "--embedding", action="store", type=str, dest="embedding")
    parser.add_option("-e", "--encoder_option", action="store", type=str, dest="encoder_option")
    parser.add_option("-l", "--load_data", action="store_true", dest="load_data")
    parser.add_option("--load_model", "--load_model", action="store_true", dest="load_model")
    parser.add_option("-o", "--optimizer", action="store", type=str, dest="optimizer")
    parser.add_option("-v", "--verbose", action="store", type=int, dest="verbose")
    parser.add_option("--highway", "--highway", action="store_true", dest="highway")
    parser.add_option("--dataset", "--dataset", action="store", type=str, dest="dataset")
    parser.add_option("--norm", "--norm_weight", action="store_true", dest="norm_weight")
    parser.add_option("--mode", "--mode", action="store", type=str, dest="mode")
    parser.add_option("--cos", "--cos", action="store_true", dest="cos")
    parser.add_option("--epochs", "--epochs", action="store", type=int, dest="epochs")
    parser.add_option("--trainable", "--trainable", action="store_true", dest="trainable")
    parser.add_option("--model_option", "--model_option", action="store", dest="model_option")
    parser.add_option("--lr", "--learning_rate", action="store", type=float, dest="learning_rate")
    parser.add_option("--ext", "--ext", action="store_true", dest="external_feat")
    return parser


def main(options):
    args = get_default_args()
    set_args(args, options)
    mode, dataset_name = args['mode'], args['dataset']

    # default setting
    args['raw_data'] = "../data/%s/" % args['dataset']
    args['qrels_file'] = "../data/%s/qrels.all.txt" % args['dataset']
    print_args(args)

    # get train/val/test names for specific dataset
    train_name, val_name, test_name, train_set, val_set, test_set, num_classes, with_url = config_dataset(args)

    max_query_len, max_doc_len, max_url_len = defaultdict(int), defaultdict(int), defaultdict(int)
    vocab = {'word': {}, '3gram': {}}
    test_vocab = {'word': {}, '3gram': {}}
    train_vocab_emb, test_vocab_emb = None, None

    ############################# LOAD DATA ##################################
    data_name = ("data_m%s_%s_%s_%s" % (mode, dataset_name, train_name, test_name)).lower()
    if args["load_data"]:
        train_dataset, vocab, train_vocab_emb, max_query_len, max_doc_len, max_url_len = load_data(
            "%s/%s/%s" % (args["experimental_data"], data_name, train_name), True)
        test_dataset, test_vocab, test_vocab_emb, _, _, _ = load_data(
            "%s/%s/%s" % (args["experimental_data"], data_name, test_name), False)
        if dataset_name != 'twitter' and dataset_name != 'TwitterURL':
            val_dataset, _, _, _, _, _ = load_data(
                "%s/%s/%s" % (args["experimental_data"], data_name, val_name), False)
        if args['embedding'] == 'glove':
            train_vocab_emb, test_vocab_emb = construct_vocab_emb(
                "%s/%s" % (args["experimental_data"], data_name), vocab['word'], test_vocab['word'], 300, "word",
                base_embed_path=args["base_embed_path"], type=args["embedding"])
        print('load dataset successfully')
    else:
        train_dataset = gen_data(args["raw_data"], train_set, vocab, test_vocab, True, max_query_len, max_doc_len,
                                 max_url_len, num_classes, args)
        print("create training set successfully...")
        if dataset_name != 'twitter' and dataset_name != 'TwitterURL':
            val_dataset = gen_data(args["raw_data"], val_set, vocab, test_vocab, False, max_query_len, max_doc_len,
                                   max_url_len, num_classes, args)
            print("create validation set successfully...")

        test_dataset = gen_data(args["raw_data"], test_set, vocab, test_vocab, False, max_query_len, max_doc_len,
                                max_url_len, num_classes, args)
        train_vocab_emb, test_vocab_emb = construct_vocab_emb(
            "%s/%s" % (args["experimental_data"], data_name), vocab['word'], test_vocab['word'], 300, "word",
            base_embed_path=args["base_embed_path"])
        save_data("%s/%s/%s" % (args["experimental_data"], data_name, train_name), True, train_dataset, max_query_len,
                  max_doc_len, max_url_len, vocab, train_vocab_emb)
        print("save training set successfully...")
        if dataset_name != 'twitter' and dataset_name != 'TwitterURL':
            save_data("%s/%s/%s" % (args["experimental_data"], data_name, val_name), False, val_dataset,
                  vocab=test_vocab, vocab_emb=test_vocab_emb)
            print("save val set successfully...")
        save_data("%s/%s/%s" % (args["experimental_data"], data_name, test_name), False, test_dataset,
                  vocab=test_vocab, vocab_emb=test_vocab_emb)
        print("save test set successfully...")

    if dataset_name == 'twitter' or dataset_name == 'TwitterURL':
        val_split = args['val_split']
        num_samples, _ = train_dataset["query_word_input"].shape
        # randomly sample queries and all their documents if query_random is True
        # otherwise, query-doc pairs are randomly sampled
        query_random = True if dataset_name == 'twitter' else False
        if query_random:
            del train_dataset["overlap_feat"]
            val_indices = sample_aaai_val_set(args["raw_data"], train_set, val_split)
        else:
            val_split = 0.1
            val_indices, val_set = [], set()
            for i in range(int(num_samples * val_split)):
                val_index = np.random.randint(num_samples)
                while val_index in val_set:
                    val_index = np.random.randint(num_samples)
                val_indices.append(val_index)
                val_set.add(val_index)

        val_dataset = {}
        for key in train_dataset:
            #print(key, train_dataset[key].shape)
            val_dataset[key] = train_dataset[key][val_indices]
            train_dataset[key] = np.delete(train_dataset[key], val_indices, 0)

    # shuffle the train dataset explicitly to make results reproducible
    # whether the performance will be affected remains a question
    keys, values = [], []
    for key in train_dataset:
        if train_dataset[key].size == 0:
            continue
        keys.append(key)
        values.append(train_dataset[key])
    zipped_values = list(zip(*values))
    random.shuffle(zipped_values)
    shuffled_values = list(zip(*zipped_values))
    for i, key in enumerate(keys):
        train_dataset[key] = np.array(shuffled_values[i])
    print('after shuffle:', train_dataset['id'][:5], train_dataset['sim'][:5],
          train_dataset['query_word_input'][:5])

    # merge the vocabulory of train and test set
    merged_vocab = {}
    merged_vocab['word'] = merge_two_dicts(vocab['word'], test_vocab['word'])
    merged_vocab['3gram'] = merge_two_dicts(vocab['3gram'], test_vocab['3gram'])
    print("TRAIN vocab: word(%d) 3gram(%d)" % (len(vocab['word']), len(vocab['3gram'])))
    print("TEST vocab: word(%d) 3gram(%d)" % (len(test_vocab['word']), len(test_vocab['3gram'])))
    print("MERGED vocab: word(%d) 3gram(%d)" % (len(merged_vocab['word']), len(merged_vocab['3gram'])))

    vocab_inv, vocab_size = {}, {}
    for key in vocab:
        vocab_inv[key] = invert_dict(merged_vocab[key])
        vocab_size[key] = len(vocab[key])
    print(vocab_size)

    # Print data samples for debug purpose
    print_dataset(mode, train_dataset, vocab_inv)
    print_dataset(mode, test_dataset, vocab_inv)

    ############################ TRAIN MODEL #################################
    # create model
    model = create_attention_model(max_query_len, max_doc_len, max_url_len, vocab_size, train_vocab_emb,
                                   args["nb_filters"], args["nb_layers"], embed_size=300,
                                   dropout_rate=args['dropout'], trainable=args["trainable"],
                                   weighting=args['weighting'], mask=args["mask"], conv_option=args['conv_option'],
                                   model_option=args['model_option'], join=args['join'],
                                   num_classes=num_classes, with_url=with_url, highway=args['highway'],
                                   att=args['co_attention'], ext_feat=args["external_feat"],
                                   encoder_option=args['encoder_option'])
    model_name = ("model_N%s_data%s_mo%s_e%s_c%s_NumFilter%d_nblayer%d_T%s_D%.1f_W%s_M%s_B%d_Val%.2f_Join%s_H%s_Att%s" % (
        mode, train_name, args['model_option'], args["encoder_option"], args['conv_option'], args["nb_filters"],
        args["nb_layers"], args["trainable"], args['dropout'], args['weighting'], args['mask'],
        args['batch_size'], args['val_split'],args['join'], args['highway'], args['co_attention'])).lower()
    model_path = "%s/%s/%s" % (args['experimental_data'], data_name, model_name)
    print(model_path)

    if args['optimizer'] == "adam":
        opt = optimizers.Adam(lr=args["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        print('use Adam optimizer')
    elif args['optimizer'] == "sgd":
        opt = optimizers.SGD(lr=args["learning_rate"], decay=1e-6, momentum=0.9, nesterov=True)
        print('use SGD optimizer')
    elif args['optimizer'] == 'rmsprop':
        opt = optimizers.RMSprop(lr=args["learning_rate"], rho=0.9, epsilon=None, decay=0.0)
        print('use RMSprop optimizer')

    if num_classes <= 2:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    else:
        print('compile model with categorical cross-entropy')
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    class_weight = None
    if args['dataset'] == 'Quora':
        #class_weight = {0:1, 1:2}
        print('apply class weight:', class_weight)

    print(model.summary())
    print('model init weights sum: %.4f' % get_model_weights(model))
    if not args['load_model']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        checkpoint = ModelCheckpoint(filepath=model_path + ".best.weights",
                                     monitor='val_loss', save_best_only=True, verbose=1)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
        model.fit(train_dataset, train_dataset['sim'], #validation_split=0.05,
                  batch_size=args['batch_size'],
                  validation_data=(val_dataset, val_dataset['sim']),
                  epochs=args['epochs'], shuffle=False,
                  callbacks=[checkpoint, lr_reducer, early_stopping],
                  class_weight=class_weight,
                  verbose=args['verbose'])


    ############################ TEST MODEL #################################
    print('load best model from %s.best.weights' % model_path)
    model.load_weights("%s.best.weights" % model_path)
    # load trained vocab embedding.
    trained_vocab_emb = model.get_layer('word-embedding').get_weights()[0]
    # merge trained vocab embedding with test OOV word embeddings
    merged_vocab_emb = np.zeros(shape=(len(merged_vocab['word']), 300))
    merged_vocab_emb[0:len(vocab['word']), :] = trained_vocab_emb
    merged_vocab_emb[len(vocab['word']):len(merged_vocab['word']), :] = test_vocab_emb
    for key in vocab:
        vocab_size[key] = len(merged_vocab[key])
    print(vocab_size)

    new_model = create_attention_model(max_query_len, max_doc_len, max_url_len, vocab_size, merged_vocab_emb,
                                       args["nb_filters"], args["nb_layers"], embed_size=300,
                                       dropout_rate=args['dropout'], trainable=args["trainable"],
                                       weighting=args['weighting'], mask=args["mask"],
                                       conv_option=args['conv_option'],
                                       model_option=args['model_option'], join=args['join'],
                                       num_classes=num_classes, with_url=with_url, highway=args['highway'],
                                       att=args['co_attention'], ext_feat=args["external_feat"],
                                       encoder_option=args['encoder_option'])
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(new_model.summary())
    for layer_id in range(len(model.layers)):
        layer = model.layers[layer_id]
        if layer.name != 'word-embedding':
            new_model.layers[layer_id].set_weights(layer.get_weights())
    print('copy weight done.')
    val_predictions = new_model.predict(val_dataset)
    predictions = new_model.predict(test_dataset)

    if dataset_name == 'twitter' or dataset_name == 'TrecQA':
        val_predictions = val_predictions[:, 1]
        predictions = predictions[:, 1]
        print(predictions[:10])
        predictions_file = "%s/%s/predictions_%s.txt" % (args["experimental_data"], data_name, model_name)
        with open(predictions_file, 'w') as f:
            for i in range(test_dataset['id'].shape[0]):
                f.write("%s %.4f %s\n" % (test_dataset['id'][i], predictions[i], args['mode']))
        print('write predictions with trec format to %s' % predictions_file)
        val_predictions_file = "%s/%s/val_predictions_%s.txt" % (args["experimental_data"], data_name, model_name)
        with open(val_predictions_file, 'w') as f:
            for i in range(val_dataset['id'].shape[0]):
                f.write("%s %.4f %s\n" % (val_dataset['id'][i], val_predictions[i], args['mode']))
        map, mrr, p30 = evaluate(val_predictions_file, args["qrels_file"])
        print('write val predictions with trec format to %s' % val_predictions_file)
        print('Validation MAP: %.4f P30: %.4f MRR: %.4f' % (map, p30, mrr))
        map, mrr, p30 = evaluate(predictions_file, args["qrels_file"])
        print('MAP: %.4f P30: %.4f MRR: %.4f' % (map, p30, mrr))
    else:
        preds = np.argmax(predictions, axis=-1)
        labels = np.argmax(test_dataset['sim'], axis=-1)
        corrects = preds == labels
        predictions_file = "%s/%s/predictions_%s.txt" % (args["experimental_data"], data_name, model_name)
        with open(predictions_file, 'w') as f:
            f.write("id label pred prob model\n")
            for i in range(len(preds)):
                f.write("%s %s %s %.4f %s\n" % (test_dataset['id'][i], labels[i],
                                                 preds[i], predictions[i][preds[i]], args['mode']))
        print('write predictions with trec format to %s' % predictions_file)
        val_preds = np.argmax(val_predictions, axis=-1)
        val_labels = np.argmax(val_dataset['sim'], axis=-1)
        val_corrects = val_preds == val_labels
        val_predictions_file = "%s/%s/val_predictions_%s.txt" % (args["experimental_data"], data_name, model_name)
        with open(val_predictions_file, 'w') as f:
            for i in range(val_dataset['id'].shape[0]):
                f.write("%s %s %s %.4f %s\n" % (val_dataset['id'][i], val_labels[i], val_preds[i],
                                                val_predictions[i][val_preds[i]], args['mode']))
        print('write val predictions with trec format to %s' % val_predictions_file)

        print('val accuracy: %.4f' % (np.count_nonzero(val_corrects)*1.0/len(val_preds)))
        print('accuracy: %.4f' % (np.count_nonzero(corrects)*1.0/len(preds)))
        macro_prec = precision_score(labels, preds, average="macro")
        macro_recall = recall_score(labels, preds, average="macro")
        print('Macro Precision: %.3f, Recall: %.3f, F1: %.3f' % (macro_prec, macro_recall,
                                                                 2*macro_prec*macro_recall/(macro_prec+macro_recall)))
        print('Micro Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision_score(labels, preds, average="micro"),
                                                                 recall_score(labels, preds, average="micro"),
                                                                 f1_score(labels, preds, average="micro")))
        print('Confusion matrix:', confusion_matrix(labels, preds))


if __name__ == "__main__":
    parser = create_option_parser()
    options, arguments = parser.parse_args()
    main(options)
