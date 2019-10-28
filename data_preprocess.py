import glob
import json
import os
import codecs
import numpy as np
import time
import sys
import re

from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
import gensim

from utils import split_sent, normalize_unicode, get_word_vector, unsplit_query, invert_dict, merge_two_dicts
from attention_model import DSSM_NUM_NEGS, ATTENTION_DEEP_LEVEL

PAD_WORD_INDEX = 0
OOV_WORD_INDEX = 1
MAX_WORD_LENGTH = 100
MAX_3GRAM_LENGTH = 200
MAX_URL_LENGTH = 120
DEFAULT_URL = "http://www.abc.com"
ps = PorterStemmer()


def read_sentences(path, vocab, is_train, repr="word", ngram_size=3, test_vocab=None):
    questions = []
    max_len = 0
    with codecs.open(path, "r", "UTF-8") as f:    
        for i, line in enumerate(f):
            q_tokens = split_sent(normalize_unicode(line.strip().lower()), repr, ngram_size)
            #digit_reg = r' [0-9]+(\.,)?[0-9]*'
            #line = re.sub(digit_reg, ' digit_regex ', line.strip().lower())
            #q_tokens = split_sent(normalize_unicode(line), repr, ngram_size)
            token_ids = []
            if len(q_tokens) > max_len:
                max_len = len(q_tokens)
            for token in q_tokens:
                if token not in vocab[repr]:
                    if is_train:
                        vocab[repr][token] = len(vocab[repr])
                    elif repr == "word" and token not in test_vocab[repr]:
                        test_vocab[repr][token] = len(vocab[repr]) + len(test_vocab[repr])
                if token in vocab[repr]:
                    token_ids.append(vocab[repr][token])
                elif repr == "word":
                    token_ids.append(test_vocab[repr][token])
                else:
                    token_ids.append(OOV_WORD_INDEX)
            questions.append(token_ids)
    return questions, max_len


def read_urls(path, vocab, is_train, repr="3gram", ngram_size=3):
    urls = []
    max_len = 0
    if os.path.exists(path):
        with codecs.open(path, "r", "UTF-8") as f:
            for i, line in enumerate(f):
                line = line.strip().lower()
                if len(line) == 0:
                    line = DEFAULT_URL
                marker_index = line.find('?')
                url = line[7:marker_index]
                if sys.version_info[0] < 3:
                    q_tokens = split_sent(normalize_unicode(unicode(url)), repr, ngram_size)
                else:
                    q_tokens = split_sent(normalize_unicode(str(url)), repr, ngram_size)
                token_ids = []
                if len(q_tokens) > max_len:
                    max_len = len(q_tokens)
                for token in q_tokens:
                    if token not in vocab[repr]:
                        if is_train:
                            vocab[repr][token] = len(vocab[repr])
                    if token in vocab[repr]:
                        token_ids.append(vocab[repr][token])
                    else:
                        token_ids.append(OOV_WORD_INDEX)
                #print(line, q_tokens, token_ids)
                urls.append(token_ids)
    return urls, max_len


def read_features(path):
    feats = []
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f):
                groups = line.strip().split()
                feats.append([float(feat) for feat in groups])
    return feats


def read_metadata(path):
    ids = []
    with open(path) as f:
        for i, line in enumerate(f):
            groups = line.strip().split()
            ids.append(" ".join(groups[:4]))
    return ids


def read_relevance(path):
    sims = []
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f):
                sims.append(int(line.strip()))
    return sims

# inject unigram and bigram IDF weight
def inject_word_weight(query_word_input, vocab_inv, weights):
    num_samples, max_query_len = query_word_input.shape
    # default unigram/bigram IDF weight as 1.0
    query_word_weight = np.ones((num_samples, ATTENTION_DEEP_LEVEL, max_query_len))
    unigram_found, bigram_found, unigram_total = 0.0, 0.0, 0.0
    for i in range(num_samples):
        for j in range(max_query_len):
            # inject word weight
            #stem_word = ps.stem(vocab_inv[query_word_input[i][j]])
            word = vocab_inv[query_word_input[i][j]]
            unigram_total += 1
            if word in weights['unigram']:
                unigram_found += 1
                query_word_weight[i][0][j] = weights['unigram'][word]

            # inject bigram weight
            if j > 0 and query_word_input[i][j-1] != PAD_WORD_INDEX:
                if query_word_input[i][j] == PAD_WORD_INDEX:
                    unigram = vocab_inv[query_word_input[i][j-1]]
                    if unigram in weights['unigram']:
                        bigram_found += 1
                        query_word_weight[i][1][j] = weights['unigram'][unigram]
                else:
                    bigram = "%s %s" % (vocab_inv[query_word_input[i][j-1]],
                                        vocab_inv[query_word_input[i][j]])
                    if bigram in weights['bigram']:
                        bigram_found += 1
                        query_word_weight[i][1][j] = weights['bigram'][bigram]
            elif j == 0:
                unigram = vocab_inv[query_word_input[i][j]]
                if unigram in weights['unigram']:
                    bigram_found += 1
                    query_word_weight[i][1][j] = weights['unigram'][unigram]

    print("Unigram found/total: %.3f (%d/%d)" %
          (unigram_found / unigram_total, unigram_found, unigram_total))
    print("Bigram found/total: %.3f (%d/%d)" %
          (bigram_found / unigram_total, bigram_found, unigram_total))
    return query_word_weight


# inject character-level ngram weight
def inject_ngram_weight(query_ngram_input, vocab_inv, weights):
    num_samples, max_query_len = query_ngram_input.shape
    query_ngram_weight = np.zeros((num_samples, ATTENTION_DEEP_LEVEL, max_query_len))
    for i in range(num_samples):
        for j in range(max_query_len):
            trigram = vocab_inv[query_ngram_input[i][j]]
            if trigram in weights['3gram']:
                query_ngram_weight[i][0][j] = weights['3gram'][trigram]

            for k in range(1, ATTENTION_DEEP_LEVEL):
                # a hard-coded 3 is because the moving window length of CNN is 3
                if j-3*k >= 0 and query_ngram_input[i][j-3*k] != PAD_WORD_INDEX:
                    ngram = ''
                    for t in range(j-3*k, j+1, 3):
                        ngram = ngram + vocab_inv[query_ngram_input[i][t]]
                    if "%dgram" % (3*(k+1)) in weights and ngram in weights["%dgram" % (3*(k+1))]:
                        query_ngram_weight[i][k][j] = weights["%dgram" % (3*(k+1))][ngram]
    return query_ngram_weight


def compute_overlap_feat(query_word_input, doc_word_input, vocab_inv, weights):
    num_samples, max_query_len = query_word_input.shape
    _, max_doc_len = doc_word_input.shape
    overlap_feats = np.zeros((num_samples, 4)) # read overlap feature
    for i in range(num_samples):
        match, idf_match = 0.0, 0.0
        bigram_match, bigram_idf_match = 0.0, 0.0
        total_word, total_idf, total_bigram_idf = 0.0, 0.0, 0.0
        for j in range(max_query_len):
            word = vocab_inv[query_word_input[i][j]]
            bigram = None
            if j > 0 and query_word_input[i][j] != PAD_WORD_INDEX:
                bigram = "%s %s" % (vocab_inv[query_word_input[i][j-1]], vocab_inv[query_word_input[i][j]])
                total_bigram_idf += float(weights['bigram'][bigram]) if bigram in weights['bigram'] else 1.0
            for k in range(max_doc_len):
                word2 = vocab_inv[doc_word_input[i][k]]
                if query_word_input[i][j] != PAD_WORD_INDEX and query_word_input[i][j] != OOV_WORD_INDEX \
                    and word == word2:
                    match += 1.0
                    idf_match += float(weights['unigram'][word]) if word in weights['unigram'] else 1.0
                    break
            for k in range(max_doc_len):
                bigram2 = None
                if k > 0 and doc_word_input[i][k] != PAD_WORD_INDEX:
                    bigram2 = "%s %s" % (vocab_inv[doc_word_input[i][k-1]], vocab_inv[doc_word_input[i][k]])
                if bigram != None and bigram == bigram2:
                    bigram_match += 1.0
                    bigram_idf_match += float(weights['bigram'][bigram]) if bigram in weights['bigram'] else 1.0
                    break

            if query_word_input[i][j] != PAD_WORD_INDEX and query_word_input[i][j] != OOV_WORD_INDEX:
                total_word += 1.0
                total_idf += float(weights['unigram'][word]) if word in weights['unigram'] else 1.0
        overlap_feats[i][0] = match / total_word
        overlap_feats[i][1] = idf_match / total_idf
        overlap_feats[i][2] = bigram_match / max(total_word-1, 1)
        overlap_feats[i][3] = bigram_idf_match / max(total_bigram_idf, 0.01)
    return overlap_feats


def create_masks(data, args):
    num_samples, max_len = data.shape
    masks = np.ones((num_samples, max_len), dtype=np.int8)
    for i in range(num_samples):
        for j in range(max_len):
            if data[i][j] == PAD_WORD_INDEX:
                masks[i][j] = 0
    return masks


# generate data from disks to machine readable format
def gen_data(path, datasets, vocab, test_vocab, is_train, max_query_len, max_doc_len, max_url_len, nb_classes, args):
    if is_train:
        vocab['word']['PAD_WORD_INDEX'] = PAD_WORD_INDEX
        vocab['word']['OOV_WORD_INDEX'] = OOV_WORD_INDEX
        vocab['3gram']['PAD_3GRAM_INDEX'] = PAD_WORD_INDEX
        vocab['3gram']['OOV_3GRAM_INDEX'] = OOV_WORD_INDEX
    query_word_list, doc_word_list, query_3gram_list, doc_3gram_list = [], [], [], []
    all_url_list, all_ids_list, all_sim_list = [], [], []
    for data_name in datasets: # there can be multiple data sets combined as the train or test data
        data_folder = "%s/%s" % (path, data_name)
        print('load dataset %s' % data_name)
        t = time.time()
        q1_word_list, max_q1_word_len = read_sentences("%s/a.toks" % data_folder, vocab, is_train,
                                                       "word", test_vocab=test_vocab)
        q2_word_list, max_q2_word_len = read_sentences("%s/b.toks" % data_folder, vocab, is_train,
                                                       "word", test_vocab=test_vocab)
        q1_3gram_list, max_q1_3gram_len = read_sentences("%s/a.toks" % data_folder, vocab, is_train,
                                                         "3gram", test_vocab=test_vocab)
        q2_3gram_list, max_q2_3gram_len = read_sentences("%s/b.toks" % data_folder, vocab, is_train,
                                                         "3gram", test_vocab=test_vocab)
        url_list = [], max_url_len_dataset = 0
        if os.path.exists("%s/url.txt" % data_folder):
            url_list, max_url_len_dataset = read_urls("%s/url.txt" % data_folder, vocab, is_train, '3gram')
        ids_list = read_metadata("%s/id.txt" % data_folder)
        if is_train:
            max_query_len['word'] = max(max_query_len['word'], min(max_q1_word_len, MAX_WORD_LENGTH))
            max_query_len['3gram'] = max(max_query_len['3gram'], min(max_q1_3gram_len, MAX_3GRAM_LENGTH))
            max_doc_len['word'] = max(max_doc_len['word'], min(max_q2_word_len, MAX_WORD_LENGTH))
            max_doc_len['3gram'] = max(max_doc_len['3gram'], min(max_q2_3gram_len, MAX_3GRAM_LENGTH))
            max_url_len['url'] = max(max_url_len['url'], min(max_url_len_dataset, MAX_URL_LENGTH))
        sim_list = read_relevance("%s/sim.txt" % data_folder)
        categorical_sim_list = np.zeros((len(sim_list), nb_classes), dtype='int')
        for i, sim in enumerate(sim_list):
            categorical_sim_list[i][sim] = 1
        print(sim_list[:5], categorical_sim_list[:5])
        query_word_list.extend(q1_word_list)
        doc_word_list.extend(q2_word_list)
        query_3gram_list.extend(q1_3gram_list)
        doc_3gram_list.extend(q2_3gram_list)
        all_url_list.extend(url_list)
        all_ids_list.extend(ids_list)
        all_sim_list.extend(categorical_sim_list)
        print("q1 max_word_len: %d, q2 max_word_len: %d, len limit: (%d, %d)" %
              (max_q1_word_len, max_q2_word_len, max_query_len['word'], max_doc_len['word']))
        print("q1 max_3gram_len: %d, q2 max_3gram_len: %d, len limit: (%d, %d)" %
              (max_q1_3gram_len, max_q2_3gram_len, max_query_len['3gram'], max_doc_len['3gram']))
        print('max_url_len: %d, limit: %d' % (max_url_len_dataset, max_url_len['url']))
        print('load dataset done: %d' % (time.time()-t))

    # question padding
    data = {'sim': np.array(all_sim_list), 'id': np.array(all_ids_list)}
    data['query_word_input'] = pad_sequences(query_word_list, maxlen=max_query_len['word'],
                                             value=PAD_WORD_INDEX, padding='post', truncating='post')
    data['query_word_mask'] = create_masks(data['query_word_input'], args)
    data['doc_word_input'] = pad_sequences(doc_word_list, maxlen=max_doc_len['word'],
                                           value=PAD_WORD_INDEX, padding='post', truncating='post')
    data['doc_word_mask'] = create_masks(data['doc_word_input'], args)
    data['query_3gram_input'] = pad_sequences(query_3gram_list, maxlen=max_query_len['3gram'],
                                              value=PAD_WORD_INDEX, padding='post', truncating='post')
    data['query_3gram_mask'] = create_masks(data['query_3gram_input'], args)
    data['doc_3gram_input'] = pad_sequences(doc_3gram_list, maxlen=max_doc_len['3gram'],
                                            value=PAD_WORD_INDEX, padding='post', truncating='post')
    data['doc_3gram_mask'] = create_masks(data['doc_3gram_input'], args)
    data['url_3gram_input'] = pad_sequences(all_url_list, maxlen=max_url_len['url'],
                                            value=PAD_WORD_INDEX, padding='post', truncating='pre')
    data['url_3gram_mask'] = create_masks(data['url_3gram_input'], args)

    if os.path.exists("%s/collection_ngram_idf.json" % path):
        t = time.time()
        weights = json.load(open("%s/collection_ngram_idf.json" % path, "r"))
        vocab_inv = invert_dict(vocab['3gram'])
        data['query_3gram_weight'] = inject_ngram_weight(data['query_3gram_input'], vocab_inv, weights)
        data['doc_3gram_weight'] = inject_ngram_weight(data['doc_3gram_input'], vocab_inv, weights)
        data['url_3gram_weight'] = inject_ngram_weight(data['url_3gram_input'], vocab_inv, weights)
        print('ngram weight injection done: %d' % (time.time()-t))

    if os.path.exists("%s/collection_word_idf.json" % path):
        t = time.time()
        weights = json.load(open("%s/collection_word_idf.json" % path, "r"))
        merge_vocab = merge_two_dicts(vocab['word'], test_vocab['word'])
        vocab_inv = invert_dict(merge_vocab)
        data['query_word_weight'] = inject_word_weight(data['query_word_input'], vocab_inv, weights)
        data['doc_word_weight'] = inject_word_weight(data['doc_word_input'], vocab_inv, weights)
        data['overlap_feat'] = compute_overlap_feat(data['query_word_input'], data['doc_word_input'],
                                                    vocab_inv, weights)
        print('word weight injection done: %d' % (time.time() - t))

    return data


def construct_vocab_emb(data_path, train_vocab, test_vocab, embed_size, qrepr, base_embed_path, type='word2vec'):
    train_vocab_emb, test_vocab_emb = None, None
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    f = open('%s/OOV_words.txt' % data_path, 'w')

    print('Load %s word embeddings...' % type)
    if type == 'word2vec':
        assert base_embed_path.endswith("GoogleNews-vectors-negative300.bin.gz")
        entity_model = gensim.models.KeyedVectors.load_word2vec_format(base_embed_path, binary=True,
                                                                       unicode_errors="ignore")
    elif type == 'glove':
        entity_model = gensim.models.KeyedVectors.load_word2vec_format(base_embed_path, binary=False,
                                                                       unicode_errors="ignore")
    else:
        base_emb_vocab = json.load(open(base_embed_path.replace("_emb", "").replace(".npy", ".json")))
        base_emb_matrix = np.load(base_embed_path)
        entity_model = (base_emb_vocab, base_emb_matrix)
    print("Building embedding matrix from base embedding at %s..." % base_embed_path)
    cnt_oov = 0
    train_vocab_emb = np.zeros(shape=(len(train_vocab), embed_size))
    test_vocab_emb = np.zeros(shape=(len(test_vocab), embed_size))
    print("train vocab size: %d, test vocab size: %d" % (len(train_vocab), len(test_vocab)))
    for word in train_vocab:
        wid = train_vocab[word]
        # padded words embedded to vector with all zeros
        if wid != PAD_WORD_INDEX:
            emb = get_word_vector(entity_model, word)
            if emb is None:
                cnt_oov += 1
                emb = np.random.rand(embed_size).astype(np.float32)
                emb = emb * 0.1
                f.write(word+'\n')
            train_vocab_emb[wid] = emb
    for word in test_vocab:
        wid = test_vocab[word] - len(train_vocab)
        emb = get_word_vector(entity_model, word)
        if emb is None:
            cnt_oov += 1
            emb = np.random.rand(embed_size).astype(np.float32)
            emb = emb * 0.1
            f.write(word+'\n')
        #print(word, test_vocab[word], wid)
        test_vocab_emb[wid] = emb

    print('OOV words: %d' % cnt_oov)
    f.close()
    return train_vocab_emb, test_vocab_emb


def load_data(folder, is_train):
    data = {}
    for fname in glob.glob("%s/*data.npy" % folder):
        key = fname.split("/")[-1].split(".")[-3]
        data[key] = np.load(fname)
    max_query_len, max_doc_len, max_url_len = None, None, None
    vocab = json.load(open("%s/vocab.json" % folder, "r"))
    vocab_emb = np.load("%s/vocab_emb.npy" % folder)
    if is_train:
        max_query_len = json.load(open("%s/max_query_len.json" % folder, "r"))
        max_doc_len = json.load(open("%s/max_doc_len.json" % folder, "r"))
        max_url_len = json.load(open("%s/max_url_len.json" % folder, "r"))
    return data, vocab, vocab_emb, max_query_len, max_doc_len, max_url_len,


def save_data(folder, is_train, data, max_query_len=None, max_doc_len=None, max_url_len=None,
              vocab=None, vocab_emb=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for key, value in data.items():
        np.save("%s/%s.data.npy" % (folder, key), value)
    json.dump(vocab, open("%s/vocab.json" % folder, "w"), indent=4)
    np.save("%s/vocab_emb" % folder, vocab_emb)
    if is_train:
        json.dump(max_query_len, open("%s/max_query_len.json" % folder, "w"))
        json.dump(max_doc_len, open("%s/max_doc_len.json" % folder, "w"))
        json.dump(max_url_len, open("%s/max_url_len.json" % folder, "w"))
