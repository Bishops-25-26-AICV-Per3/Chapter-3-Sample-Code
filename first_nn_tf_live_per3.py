import tensorflow as tf

INPUT_SHAPE = (224, 224, 3)

class Model():
    """Represent the CNN as an object."""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape is expected to be channels last."""
        self.model = tf.keras.Sequential()
        # The first layer has to have input_shape as a keyword argument
        self.model.add(tf.keras.layers.ZeroPadding2D(((2, 1), (2, 1)), 
            input_shape = input_shape))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 48, # output channels
            kernel_size = 11,
            strides = 4,
        ))

        ... # More convolution blocks here

        self.model.add(tf.keras.layers.Flatten())
        # Dense layers are also called Linear or Fully-Connected layers
        # This section is also called the MLP = Multi-Layer Perceptron

        # Input to the constructor is number of outputs
        self.model.add(tf.keras.layers.Dense(2048))


def main():
    model = Model(INPUT_SHAPE)
    model.model.summary()

if __name__ == "__main__":
    main()
