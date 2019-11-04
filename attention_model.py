from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    Lambda, TimeDistributed, SpatialDropout1D, Reshape, RepeatVector, Bidirectional, LSTM
from keras.layers.merge import Dot, Concatenate, Multiply, Add
from keras.layers.advanced_activations import Softmax
from keras.models import Model
from keras import backend as K
from keras.initializers import RandomUniform
from bi_attention_layer import BiAttentionLayer
from highway_layer import HighwayLayer
from encoders import add_deep_conv_layer, add_wide_conv_layer, add_bilstm_layer


ATTENTION_DEEP_LEVEL = 5

def keras_diagonal(x):
    seq_len = 10 if 'seq_len' not in K.params else K.params['seq_len']
    if seq_len != 10: print("dim of diagonal matrix in keras_diagonal: %d" % seq_len)
    return K.sum(K.eye(seq_len) * x, axis=1)

def elementwise_prod(x, y):
    (_, vec_size,) = K.int_shape(x)
    attention_level = K.params['attention_level']
    return x*y[:, attention_level, 0:vec_size]

def attention_weighting_prod(attention, weights):
    attention_level = K.params['attention_level']
    return attention * weights[:, attention_level, :, :]

def repeat_vector(x, rep, axis):
    return K.repeat(x, rep, axis)


def max_pooling(x):
    return K.max(x, axis=1)

def mean_pooling(x):
    return K.mean(x, axis=1)

def max_pooling_with_mask(x, query_mask):
    # x is batch_size * |doc| * |query|
    # query_mask is batch_size * |query| (with masks as 0)
    return K.max(x, axis=1) * query_mask

def mean_pooling_with_mask(x, doc_mask, query_mask):
    # x is batch_size * |doc| * |query|
    # doc_mask is batch_size * |doc| (with masks as 0)
    # query_mask is batch_size * |query| (with masks as 0)
    ZERO_SHIFT = 0.1
    doc_mask_sum = (K.sum(doc_mask, axis=-1, keepdims=True) + ZERO_SHIFT)
    return query_mask * K.batch_dot(x, doc_mask, axes=[1, 1]) / doc_mask_sum

def add_embed_layer(vocab_emb, vocab_size, embed_size, train_embed, dropout_rate, layer_name):
    emb_layer = Sequential(name=layer_name)
    if vocab_emb is not None:
        print("Embedding with initialized weights")
        print(vocab_size, embed_size)
        emb_layer.add(Embedding(input_dim=vocab_size, output_dim=embed_size, weights=[vocab_emb],
                                trainable=train_embed, mask_zero=False))
    else:
        print("Embedding with random weights")
        emb_layer.add(Embedding(input_dim=vocab_size, output_dim=embed_size, trainable=True, mask_zero=False,
                                embeddings_initializer=RandomUniform(-0.05, 0.05)))
    emb_layer.add(SpatialDropout1D(dropout_rate))
    return emb_layer

def add_conv_layer(input_list, layer_name, nb_filters, kernel_size, padding, dropout_rate=0.1,
                   activation='relu', strides=1, attention_level=0, conv_option="normal", prev_conv_tensors=None):
    conv_layer = Convolution1D(filters=nb_filters, kernel_size=kernel_size, padding=padding,
                               activation=activation, strides=strides, name=layer_name)
    max_pooling_layer = GlobalMaxPooling1D()
    #normlize_layer = BatchNormalization()
    dropout_layer = Dropout(dropout_rate)
    output_list, conv_output_list = [], []
    for i in range(len(input_list)):
        input = input_list[i]
        conv_tensor = conv_layer(input)
        if conv_option == "ResNet":
            conv_tensor = Add()([conv_tensor, prev_conv_tensors[i][-1]])
        #normlize_tensor = normlize_layer(conv_tensor)
        dropout_tensor = dropout_layer(conv_tensor)
        conv_pooling_tensor = max_pooling_layer(conv_tensor)
        output_list.append(dropout_tensor)
        #conv_output_list.append(conv_pooling_tensor)
        conv_output_list.append(conv_tensor)
    return output_list, conv_output_list


def add_attention_layer(query_embedding, doc_embedding, layer_name, query_mask=None, doc_mask=None, mask=False):
    dot_prod = Dot(axes=-1, name=layer_name)([doc_embedding, query_embedding])
    norm_sim = Activation('softmax')(dot_prod)
    if mask:
        max_sim = Lambda(lambda x: max_pooling_with_mask(x[0], x[1]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, query_mask])
        mean_sim = Lambda(lambda x: mean_pooling_with_mask(x[0], x[1], x[2]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, doc_mask, query_mask])
    else:
        max_sim = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2], ))(norm_sim)
        mean_sim = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)
    return norm_sim, max_sim, mean_sim

def add_attention_layer_with_query_weighting(query_embedding, doc_embedding, layer_name, attention_level,
                                             query_weight, query_mask=None, doc_mask=None, mask=False):
    """
       Dot -> softmax -> pooling -> (mask) -> weighting

    """

    dot_prod = Dot(axes=-1, name=layer_name)([doc_embedding, query_embedding])
    norm_sim = Activation('softmax')(dot_prod)
    if mask:
        max_sim = Lambda(lambda x: max_pooling_with_mask(x[0], x[1]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, query_mask])
        mean_sim = Lambda(lambda x: mean_pooling_with_mask(x[0], x[1], x[2]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, doc_mask, query_mask])
    else:
        max_sim = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2], ))(norm_sim)
        mean_sim = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)
    if attention_level <= 1:
        setattr(K, 'params', {'attention_level': attention_level})
        max_sim = Lambda(lambda x: elementwise_prod(x[0], x[1]),
                     output_shape=lambda inp_shp: (inp_shp[0][0], inp_shp[0][1],))([max_sim, query_weight])
        mean_sim = Lambda(lambda x: elementwise_prod(x[0], x[1]),
                      output_shape=lambda inp_shp: (inp_shp[0][0], inp_shp[0][1]))([mean_sim, query_weight])
    return norm_sim, max_sim, mean_sim


def add_attention_layer_with_doc_weighting(query_embedding, doc_embedding, layer_name, attention_level,
                                           query_weight, doc_weight, max_query_len, max_doc_len,
                                           query_mask=None, doc_mask=None, mask=False, nb_layers=ATTENTION_DEEP_LEVEL):
    dot_prod = Dot(axes=-1, name=layer_name)([doc_embedding, query_embedding])
    norm_sim = Activation('softmax')(dot_prod)
    reshaped_query_weight = Reshape((nb_layers, max_query_len, 1),
                                    input_shape=(nb_layers, max_query_len,))(query_weight)
    repeated_query_weight = Lambda(lambda x: repeat_vector(x[0], x[1], x[2]), output_shape=lambda inp_shp:(
        inp_shp[0][0], inp_shp[0][1], max_query_len, max_doc_len,))([reshaped_query_weight, max_doc_len, -1])
    reshaped_doc_weight = Reshape((nb_layers, 1, max_doc_len),
                                  input_shape=(nb_layers, max_doc_len,))(doc_weight)
    repeated_doc_weight = Lambda(lambda x: repeat_vector(x[0], x[1], x[2]), output_shape=lambda inp_shp: (
        inp_shp[0][0], inp_shp[0][1], max_query_len, max_doc_len,))([reshaped_doc_weight, max_query_len, -2])
    weight_product = Multiply()([repeated_query_weight, repeated_doc_weight])
    transformed_weight_product = Dense(max_doc_len, activation='relu')(weight_product)
    setattr(K, 'params', {'attention_level': attention_level})
    norm_sim = Lambda(lambda x: attention_weighting_prod(x[0], x[1]), output_shape=lambda inp_shp:(
        inp_shp[0][0], inp_shp[0][1], inp_shp[0][2]))([norm_sim, transformed_weight_product])
    if mask:
        max_sim = Lambda(lambda x: max_pooling_with_mask(x[0], x[1]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, query_mask])
        mean_sim = Lambda(lambda x: mean_pooling_with_mask(x[0], x[1], x[2]), output_shape=lambda inp_shp: (
            inp_shp[0][0], inp_shp[0][2],))([norm_sim, doc_mask, query_mask])
    else:
        max_sim = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2], ))(norm_sim)
        mean_sim = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(norm_sim)
    return norm_sim, max_sim, mean_sim

def add_context2query_layer(query_embedding, biattention_matrix, attention_level, max_query_len, max_doc_len):
    # Following the context-to-query implementation of BiDAF model
    # query_embedding: batch_size * max_query_len * nb_filters
    # biattention_matrix: batch_size * max_query_len * max_doc_len
    norm_biattention = Softmax(axis=-2)(biattention_matrix)
    # Activation('softmax', axis=-2)(biattention_matrix)
    reshape_norm_biatt = Reshape((max_doc_len, max_query_len, ))(norm_biattention)
    context_embedding = Dot(axes=[-1, -2], name="context2query-%d" % attention_level)(
        [reshape_norm_biatt, query_embedding])
    return context_embedding


def add_query2context_layer(doc_embedding, biattention_matrix, attention_level, max_query_len, max_doc_len, nb_filters):
    # query_embedding: batch_size * max_doc_len * nb_filters
    # biattention_matrix: batch_size * max_query_len * max_doc_len
    max_biattention = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2], ))(biattention_matrix)
    norm_biatt = Activation('softmax')(max_biattention)
    reshape_doc_embedding = Reshape((nb_filters, max_doc_len, ))(doc_embedding)
    context_embedding = Dot(axes=-1, name="query2context-%d" % attention_level)([reshape_doc_embedding, norm_biatt])
    return context_embedding

########################## Our model implementation #########################

def create_attention_model(max_query_len, max_doc_len, max_url_len, vocab_size, embedding_matrix, nb_filters,
                           nb_layers=ATTENTION_DEEP_LEVEL, embed_size=300, dropout_rate=0.1, trainable=True,
                           weighting=False, mask=False, conv_option="normal", model_option="complete",
                           join="matching", num_classes=2, highway=True, att='BiDAF', ext_feat=False,
                           with_url=False, encoder_option='deepconv'):
    print('create attention model...')
    model = Sequential()
    query_word_input = Input(shape=(max_query_len['word'], ), name="query_word_input")
    doc_word_input = Input(shape=(max_doc_len['word'],), name="doc_word_input")
    input_list = [query_word_input, doc_word_input]
    if model_option == "complete":
        query_char_input = Input(shape=(max_query_len['3gram'],), name="query_3gram_input")
        doc_char_input = Input(shape=(max_doc_len['3gram'],), name="doc_3gram_input")
        input_list.extend([query_char_input, doc_char_input])

    if with_url:
        url_char_input = Input(shape=(max_url_len['url'],), name="url_3gram_input")
        input_list.append(url_char_input)

    # Define Mask
    query_word_mask, query_char_mask, doc_word_mask, doc_char_mask, url_char_mask = None, None, None, None, None
    if mask:
        query_word_mask = Input(shape=(max_query_len['word'], ), name="query_word_mask")
        doc_word_mask = Input(shape=(max_doc_len['word'], ), name="doc_word_mask")
        query_char_mask = Input(shape=(max_query_len['3gram'],), name="query_3gram_mask")
        doc_char_mask = Input(shape=(max_doc_len['3gram'],), name="doc_3gram_mask")
        if with_url: url_char_mask = Input(shape=(max_url_len['url'],), name="url_3gram_mask")
        input_list.extend([query_word_mask, doc_word_mask, query_char_mask, url_char_mask])

    query_word_weight, query_char_weight, doc_word_weight, doc_char_weight = None, None, None, None
    external_feat, url_char_weight = None, None
    if weighting == 'query':
        query_word_weight = Input(shape=(nb_layers, max_query_len['word'], ), name="query_word_weight")
        query_char_weight = Input(shape=(nb_layers, max_query_len['3gram'],), name="query_3gram_weight")
        input_list.extend([query_word_weight, query_char_weight])
    elif weighting == 'doc':
        query_word_weight = Input(shape=(nb_layers, max_query_len['word'],), name="query_word_weight")
        query_char_weight = Input(shape=(nb_layers, max_query_len['3gram'],), name="query_3gram_weight")
        doc_word_weight = Input(shape=(nb_layers, max_doc_len['word'],), name="doc_word_weight")
        doc_char_weight = Input(shape=(nb_layers, max_doc_len['3gram'],), name="doc_3gram_weight")
        input_list.extend([query_word_weight, query_char_weight, doc_word_weight, doc_char_weight])
        if with_url:
            url_char_weight = Input(shape=(nb_layers, max_url_len['url'],), name="url_3gram_weight")
            input_list.append(url_char_weight)

    if ext_feat:
        external_feat = Input(shape=(4,), name='overlap_feat')
        input_list.append(external_feat)


    # Create query-doc word-to-word attention layer
    query_word_embedding_layer = add_embed_layer(embedding_matrix, vocab_size['word'], embed_size, trainable,
                                                 dropout_rate, 'word-embedding')

    query_embedding = query_word_embedding_layer(query_word_input)
    doc_embedding = query_word_embedding_layer(doc_word_input)
    if highway:
        highway_layer = HighwayLayer()
        highway_query_layer = TimeDistributed(highway_layer, name='highway_query_layer')
        highway_doc_layer = TimeDistributed(highway_layer, name='highway_doc_layer')
        query_embedding = highway_query_layer(query_embedding)
        doc_embedding = highway_doc_layer(doc_embedding)

    norm_sim_list, max_sim_list, mean_sim_list = [], [], []
    attention_emb_list = []

    if encoder_option == "deepconv":
        output_list, conv_output_list = add_deep_conv_layer(
            query_embedding, doc_embedding, nb_layers, "deep-word-conv", nb_filters, 2, "same",
            dropout_rate, strides=1, conv_option=conv_option)
    elif encoder_option == "wideconv":
        output_list, conv_output_list = add_wide_conv_layer(
            query_embedding, doc_embedding, nb_layers, "wide-word-conv", nb_filters, 2, "same",
            dropout_rate, strides=1, conv_option=conv_option)
    elif encoder_option == "bilstm":
        output_list, conv_output_list = add_bilstm_layer(
            query_embedding, doc_embedding, nb_layers, "bilstm-word", nb_filters, dropout_rate)
    else:
        print('invalid encoder choice!')
        exit(0)

    for i in range(nb_layers):
        query_embedding, doc_embedding = output_list[i][0], output_list[i][1]
        if i > 0:
            if join == 'biattention' or join == 'hcan':
                biattention_matrix = BiAttentionLayer(nb_filters, max_query_len['word'], max_doc_len['word'],
                                                      name="bi-attention%d" % i)([query_embedding, doc_embedding])
                if att == 'BiDAF':
                    attentive_doc_embedding = add_context2query_layer(query_embedding, biattention_matrix, i,
                                                                      max_query_len['word'], max_doc_len['word'])
                    context_emb = add_query2context_layer(doc_embedding, biattention_matrix, i,
                                                          max_query_len['word'], max_doc_len['word'], nb_filters)
                    reshape_context_emb = Reshape((1, nb_filters))(context_emb)
                    rep_context_emb = Lambda(lambda x: K.repeat_elements(x, max_doc_len['word'], -2),
                                             output_shape=lambda inp_shp: (inp_shp[0], max_doc_len['word'], nb_filters,))(
                        reshape_context_emb)
                    attentive_doc_emb_prod = Lambda(lambda x: x[0] * x[1])([doc_embedding, attentive_doc_embedding])
                    context_doc_emb_prod = Lambda(lambda x: x[0] * x[1])([rep_context_emb, attentive_doc_embedding])
                    concat_doc_emb = Concatenate(axis=-1)([doc_embedding, attentive_doc_embedding,
                                                           attentive_doc_emb_prod, context_doc_emb_prod])
                    all_doc_emb = Activation('relu')(concat_doc_emb)
                    attention_emb = Bidirectional(LSTM(int(nb_filters/2), recurrent_dropout=dropout_rate, dropout=dropout_rate,
                             return_sequences=True, name="biattention-lstm%d" % i))(all_doc_emb)
                    attention_sim1 = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(attention_emb)
                    attention_sim2 = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(attention_emb)
                    attention_emb_list.extend([attention_sim1, attention_sim2])
                elif att == 'ESIM':
                    attentive_doc_embedding = add_context2query_layer(query_embedding, biattention_matrix, 2*i-1,
                                                                      max_query_len['word'], max_doc_len['word'])
                    subtract_doc_emb = Lambda(lambda x: x[0] - x[1])([attentive_doc_embedding, doc_embedding])
                    prod_doc_emb = Lambda(lambda x: x[0] * x[1])([attentive_doc_embedding, doc_embedding])
                    merge_doc_emb = Concatenate(axis=-1)([attentive_doc_embedding, doc_embedding,
                                                          subtract_doc_emb, prod_doc_emb])
                    seq_doc_emb = Bidirectional(LSTM(nb_filters, recurrent_dropout=dropout_rate, dropout=dropout_rate,
                             return_sequences=True, name="biattention-lstm%d" % (2*i-1)))(merge_doc_emb)
                    max_doc_emb = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(seq_doc_emb)
                    #mean_doc_emb = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(seq_doc_emb)

                    attentive_query_embedding = add_context2query_layer(doc_embedding, biattention_matrix, 2*i,
                                                                        max_doc_len['word'], max_query_len['word'])
                    subtract_query_emb = Lambda(lambda x: x[0]-x[1])([attentive_query_embedding, query_embedding])
                    prod_query_emb = Lambda(lambda x: x[0]*x[1])([attentive_query_embedding, query_embedding])
                    merge_query_emb = Concatenate(axis=-1)([attentive_query_embedding, query_embedding,
                                                            subtract_query_emb, prod_query_emb])
                    seq_query_emb = Bidirectional(LSTM(nb_filters, recurrent_dropout=dropout_rate, dropout=dropout_rate,
                                    return_sequences=True, name="biattention-lstm%d" % (2*i)))(merge_query_emb)
                    max_query_emb = Lambda(max_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(seq_query_emb)
                    #mean_query_emb = Lambda(mean_pooling, output_shape=lambda inp_shp: (inp_shp[0], inp_shp[2],))(seq_query_emb)
                    attention_emb_list.extend([max_query_emb, max_doc_emb])
                elif att == 'DecompAtt':
                    None
                else:
                    print('invalid attention choice!')
                    exit(0)

        if weighting == 'query':
            norm_sim, max_sim, mean_sim = add_attention_layer_with_query_weighting(
                query_embedding, doc_embedding, "word-attention%d" % i, i, query_word_weight,
                query_word_mask, doc_word_mask, mask)
        elif weighting == 'doc':
            norm_sim, max_sim, mean_sim = add_attention_layer_with_doc_weighting(
                query_embedding, doc_embedding, "word-attention%d" % i, i, query_word_weight, doc_word_weight,
                max_query_len['word'], max_doc_len['word'], query_word_mask, doc_word_mask, mask, nb_layers)
        else:
            norm_sim, max_sim, mean_sim = add_attention_layer(query_embedding, doc_embedding,
                                                              "word-attention%d" % i,
                                                              query_word_mask, doc_word_mask, mask)
        norm_sim_list.append(norm_sim)
        max_sim_list.append(max_sim)
        mean_sim_list.append(mean_sim)

    if model_option == 'complete':
        # Create query-doc char-to-char attention layer
        char_embedding_layer = add_embed_layer(None, vocab_size['3gram'], 50, True, dropout_rate, 'char-embedding')
        query_char_embedding_layer = url_char_embedding_layer = doc_char_embedding_layer = char_embedding_layer

        doc_char_embedding = doc_char_embedding_layer(doc_char_input)
        url_char_embedding = url_char_embedding_layer(url_char_input) if with_url else None

        #if highway:
        #    char_highway_layer = TimeDistributed(HighwayLayer(), name='highway_char_embedding')
        #    doc_char_embedding = char_highway_layer(doc_char_embedding)
        #    url_char_embedding = char_highway_layer(url_char_embedding)

        char_embedding_candidates = [doc_char_embedding, url_char_embedding]
        char_weight_candidates = [doc_char_weight, url_char_weight]
        char_mask_candidates = [doc_char_mask, url_char_mask]
        max_doc_len_candidates = [max_doc_len["3gram"], max_url_len["url"]]

        j = 0
        for char_embedding, char_mask, char_weight, max_doc_len_tmp in zip(char_embedding_candidates, char_mask_candidates,
                                                                           char_weight_candidates, max_doc_len_candidates):

            if char_embedding == None: continue
            query_embedding = query_char_embedding_layer(query_char_input)
            #if highway:
            #    query_embedding = char_highway_layer(query_embedding)
            conv_embedding_list = [[query_embedding], [char_embedding]]
            setattr(K, 'params', {'max_doc_len': max_doc_len_tmp, 'max_query_len': max_query_len['3gram']})

            if encoder_option == "deepconv" or encoder_option == "bilstm":
                char_output_list, char_conv_output_list = add_deep_conv_layer(
                    query_embedding, char_embedding, nb_layers, "deep-char-conv", nb_filters, 4, "same",
                    dropout_rate, strides=1, conv_option=conv_option, base_layer_idx=j)
            elif encoder_option == "wideconv":
                char_output_list, char_conv_output_list = add_wide_conv_layer(
                    query_embedding, char_embedding, nb_layers, "wide-char-conv", nb_filters, 4, "same",
                    dropout_rate, strides=1, conv_option=conv_option, base_layer_idx=j)
            else:
                print('invalid encoder choice!')
                exit(0)

            for i in range(nb_layers):
                query_embedding, char_embedding = char_output_list[i][0], char_output_list[i][1]
                query_embedding, char_embedding = char_output_list[i][0], char_output_list[i][1]
                if weighting == 'query':
                    norm_sim2, max_sim2, mean_sim2 = add_attention_layer_with_query_weighting(
                        query_embedding, char_embedding, "char-attention%d" % j, i, query_char_weight,
                        query_char_mask, char_mask, mask)
                elif weighting == 'doc':
                    norm_sim2, max_sim2, mean_sim2 = add_attention_layer_with_doc_weighting(
                        query_embedding, char_embedding, "char-attention%d" % j, i, query_char_weight, char_weight,
                        max_query_len['3gram'], max_url_len['url'], query_char_mask, char_mask, mask)
                else:
                    norm_sim2, max_sim2, mean_sim2 = add_attention_layer(query_embedding, char_embedding, "char-attention%d" % j,
                                                                         query_char_mask, char_mask, mask)

                norm_sim_list.append(norm_sim2)
                max_sim_list.append(max_sim2)
                mean_sim_list.append(mean_sim2)

                j += 1

    if join == 'matching':
        if len(max_sim_list) == 1:
            feature_vector = max_sim_list[0]
        else:
            feature_vector = Concatenate(axis=-1, name="feature_vector")(max_sim_list)
    elif join == 'biattention':
        if len(attention_emb_list) == 1:
            feature_vector = attention_emb_list[0]
        else:
            feature_vector = Concatenate(axis=-1, name="feature_vector")(attention_emb_list)
    elif join == 'hcan':
        max_sim_list.extend(attention_emb_list)
        feature_vector = Concatenate(axis=-1, name="feature_vector")(max_sim_list)
    else:
        raise Exception('invalid join method!')
    #max_sim_list.extend(mean_sim_list)
    #max_sim_list.extend(conv_embedding_list)
    #feature_vector = Concatenate(axis=-1, name="feature_vector")(max_sim_list)
    if ext_feat:
        feature_vector = Concatenate(axis=-1, name="external_feature_vector")([feature_vector, external_feat])
    #if highway:
    #    final_highway_layer = HighwayLayer(name='final_highway_layer')
    #    feature_vector = final_highway_layer(feature_vector)

    feature_vector1 = Dense(150, activation='relu', name="feature_vector1")(feature_vector)
    feature_vector2 = Dense(50, activation='relu', name="feature_vector2")(feature_vector1)
    prediction = Dense(num_classes, activation='softmax', name="prediction")(feature_vector2)
    return Model(input_list, [prediction])
