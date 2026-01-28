import tensorflow as tf

INPUT_SHAPE = (224, 224, 3)

class Model():
    """Represent the CNN as an object."""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape is expected to be channels last."""
        self.model = tf.keras.Sequential()
        # Your first layer of any type requires the input_shape keyword arg.
        self.model.add(tf.keras.layers.ZeroPadding2D(
            ((2, 1), (2, 1)),
            input_shape = input_shape,
        ))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 48,
            kernel_size = 11,
            strides = 4,
            activation = "relu",
        ))
        self.model.add(tf.keras.layers.Flatten())
        # Dense layers aka Linear or Fully-connected layers
        # The section at the end is aka as MLP = Multi-Layer Perceptron

        # The only input to the constructor is the number of output features.
        self.model.add(tf.keras.layers.Dense(2048))

def main():
    model = Model(INPUT_SHAPE)
    model.model.summary()

if __name__ == "__main__":
    main()