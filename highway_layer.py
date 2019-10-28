from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, activations
from keras.layers.merge import Multiply, Subtract

class HighwayLayer(Layer):
    def __init__(self, activation='relu', kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
        super(HighwayLayer, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.output_dim = input_dim
        self.W1 = self.add_weight(shape=(input_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=self.kernel_regularizer)
        self.W2 = self.add_weight(shape=(input_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=self.kernel_regularizer)
        self.b1 = self.add_weight(shape=(self.output_dim, ),
                                  initializer=self.bias_initializer,
                                  name='b1',
                                  regularizer=self.kernel_regularizer)
        self.b2 = self.add_weight(shape=(self.output_dim, ),
                                  initializer=self.bias_initializer,
                                  name='b2')
        self.trainable_weights = [self.W1, self.W2, self.b1, self.b2]

    def call(self, x):
        highway = self.activation(K.bias_add(K.dot(x, self.W1), self.b1))
        carry = self.activation(K.bias_add(K.dot(x, self.W2), self.b2))
        o = highway*carry + x*(1-carry)
        #o = Multiply()([highway, carry]) + Multiply()([x, 1-carry])
        return o

    def compute_output_shape(self, input_shape):
        return input_shape

    def test(self):
        from keras.layers import Input
        from keras.models import Model
        import numpy as np
        input = Input(shape=(4, ))
        output = HighwayLayer()(input)
        m = Model(input, output)
        print(m.summary())
        tensor_a = np.random.rand(2, 4)
        print(tensor_a, m.predict(tensor_a))