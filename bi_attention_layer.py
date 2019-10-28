from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers

class BiAttentionLayer(Layer):
    def __init__(self, input_dim, max_sent1, max_sent2,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(BiAttentionLayer, self).__init__(**kwargs)
        self.max_sent1 = max_sent1
        self.max_sent2 = max_sent2
        self.kernel_initializer = initializers.get(kernel_initializer)
        #self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        #self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.W1 = self.add_weight(shape=(self.input_dim, 1),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=self.kernel_regularizer)
        self.W2 = self.add_weight(shape=(self.input_dim, 1),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=self.kernel_regularizer)
        self.bilinear_weights = self.add_weight(shape=(self.input_dim, self.input_dim),
                                                initializer=self.kernel_initializer,
                                                name='bilinear_weights',
                                                regularizer=self.kernel_regularizer)
        self.trainable_weights = [self.W1, self.W2, self.bilinear_weights]

    def call(self, inputs):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('BiAttentionLayer must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        if K.ndim(inputs[0]) < 3 or K.ndim(inputs[1]) < 3:
            raise Exception('input tensors with insufficent dimensions:'
                            + str(K.shape(inputs[0])) + str(K.shape(inputs[1])))

        # s1, s2: batch_size * time_steps * input_dim
        s1, s2 = inputs[0], inputs[1]
        batch_size = K.shape(s1)[0]
        #print(K.shape(s1))
        attention1 = K.dot(s1, self.W1)
        attention2 = K.dot(s2, self.W2)
        #print(attention1, attention2)
        bilinear_attention = K.batch_dot(s1, K.dot(s2, self.bilinear_weights), axes=2)
        #print(bilinear_attention)
        rep_attention1 = K.repeat_elements(attention1, self.max_sent2, -1)
        reshape_attention2 = K.reshape(attention2, (batch_size, 1, self.max_sent2))
        rep_attention2 = K.repeat_elements(reshape_attention2, self.max_sent1, -2)
        return rep_attention1+rep_attention2+bilinear_attention

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.max_sent1, self.max_sent2)

    def test(self):
        from keras.layers import Input
        from keras.models import Model
        import numpy as np
        input_a = Input(shape=(3,4,))
        input_b = Input(shape=(5,4,))
        bi_attention = BiAttentionLayer(4,3,5)([input_a, input_b])
        m = Model([input_a, input_b], bi_attention)
        print(m.summary())
        tensor_a = np.ones((2,3,4))
        tensor_b = np.ones((2,5,4))
        m.predict([tensor_a, tensor_b])
