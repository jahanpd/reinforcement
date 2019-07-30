from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class equationNine(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(equationNine, self).__init__(**kwargs)

    def build(self, input_shape):
        super(equationNine, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        actions, state = x # a is actions and b is state value
        return (state + (actions - K.mean(actions)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)