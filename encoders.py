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


def add_deep_conv_layer(query_emb, doc_emb, nb_layers, layer_name, nb_filters, kernel_size, padding,
                        dropout_rate=0.1, activation='relu', strides=1, conv_option="normal",
                        base_layer_idx=0):
    output_list, conv_output_list = [[query_emb, doc_emb]], [[query_emb, doc_emb]]
    for i in range(nb_layers):
        conv_layer = Convolution1D(filters=nb_filters, kernel_size=kernel_size, padding=padding,
                                   activation=activation, strides=strides, name="%s-%d" % (layer_name, i+base_layer_idx))
        max_pooling_layer = GlobalMaxPooling1D()
        #normlize_layer = BatchNormalization()
        dropout_layer = Dropout(dropout_rate)
        query_conv_tensor, doc_conv_tensor = conv_layer(output_list[i][0]), conv_layer(output_list[i][1])
        if conv_option == "ResNet":
            query_conv_tensor = Add()([query_conv_tensor, conv_output_list[i][0]])
            doc_conv_tensor = Add()([doc_conv_tensor, conv_output_list[i][1]])
        query_dropout_tensor = dropout_layer(query_conv_tensor)
        doc_dropout_tensor = dropout_layer(doc_conv_tensor)
        output_list.append([query_dropout_tensor, doc_dropout_tensor])
        conv_output_list.append([query_conv_tensor, doc_conv_tensor])
    return output_list, conv_output_list


def add_wide_conv_layer(query_emb, doc_emb, nb_layers, layer_name, nb_filters, kernel_size, padding,
                        dropout_rate=0.1, activation='relu', strides=1, conv_option="normal", base_layer_idx=0):
    output_list, conv_output_list = [[query_emb, doc_emb]], []
    for i in range(nb_layers):
        conv_layer = Convolution1D(filters=nb_filters, kernel_size=(kernel_size-1)*i+kernel_size, padding=padding,
                                   activation=activation, strides=strides, name="%s-%d" % (layer_name, i+base_layer_idx))
        dropout_layer = Dropout(dropout_rate)
        query_conv_tensor, doc_conv_tensor = conv_layer(query_emb), conv_layer(doc_emb)
        query_dropout_tensor = dropout_layer(query_conv_tensor)
        doc_dropout_tensor = dropout_layer(doc_conv_tensor)
        output_list.append([query_dropout_tensor, doc_dropout_tensor])
    return output_list, conv_output_list

def add_bilstm_layer(query_emb, doc_emb, nb_layers, layer_name, nb_filters, dropout_rate=0.1):
    output_list, conv_output_list = [[query_emb, doc_emb]], []
    for i in range(nb_layers):
        bilstm_layer = Bidirectional(LSTM(int(nb_filters/2), recurrent_dropout=dropout_rate, dropout=dropout_rate,
                                          return_sequences=True, name="%s-%d" % (layer_name, i)))
        query_lstm_tensor, doc_lstm_tensor = bilstm_layer(output_list[i][0]), bilstm_layer(output_list[i][1])
        output_list.append([query_lstm_tensor, doc_lstm_tensor])
    return output_list, conv_output_list
