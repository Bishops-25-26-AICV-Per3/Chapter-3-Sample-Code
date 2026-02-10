import os

import tensorflow as tf

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
# Supresses some informational messages & warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

class Model():
    """Represent the CNN as an object."""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape is expected to be channels last."""
        self.model = tf.keras.Sequential()
        # Small correction: Now use InputLayer instead of input_shape kwarg
        self.model.add(tf.keras.layers.InputLayer(shape = input_shape))
        self.model.add(tf.keras.layers.ZeroPadding2D(((2, 1), (2, 1))))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 48, # output channels
            kernel_size = 11,
            strides = 4,
        ))
        self.model.add(tf.keras.layers.ZeroPadding2D(2))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = 5,
            strides = 1,
            activation = 'relu',
        ))
        self.model.add(tf.keras.layers.MaxPool2D(
            pool_size = 3,
            strides = 2,
        ))
        self.model.add(tf.keras.layers.ZeroPadding2D(1))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 192,
            kernel_size = 3,
            strides = 1,
            activation = 'relu',
        ))
        self.model.add(tf.keras.layers.MaxPool2D(
            pool_size = 3,
            strides = 2,
        ))
        self.model.add(tf.keras.layers.ZeroPadding2D(1))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 192,
            kernel_size = 3,
            strides = 1,
            activation = 'relu',
        ))
        self.model.add(tf.keras.layers.ZeroPadding2D(1))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = 3,
            strides = 1,
            activation = 'relu',
        ))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(2048))
        self.model.add(tf.keras.layers.Dense(2048))
        self.model.add(tf.keras.layers.Dense(1024))
        self.model.add(tf.keras.layers.Dense(3))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = ['accuracy']
        )

def main():
    train, validation = tf.keras.utils.image_dataset_from_directory(
        "../animals",
        label_mode = "categorical",
        batch_size = BATCH_SIZE,
        image_size = (224, 224),
        seed = 37,
        validation_split = 0.2, 
        subset = "both",
    )

    augmented = train.map(lambda x, y: (tf.image.flip_left_right(x), y))
    train = train.concatenate(augmented)
    train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    validation = validation.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    model = Model(INPUT_SHAPE)
    model.model.summary()

    model.model.fit(
        train,
        validation_data = validation,
        epochs = 20,
        verbose = 1,
    )

if __name__ == "__main__":
    main()
