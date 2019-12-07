from preprocess import *
import tensorflow as tf
import glob
import wandb

# Set hyper-parameters
#wandb.init()

def train():
    import numpy as np
    import tensorflow as tf
    import wandb

    config_defaults = {
        'conv1': 32,
        'conv2': 32,
        'conv3': 10,
        'dense1': 8,
        'dense2': 8,
    }
    wandb.init(config=config_defaults)

    config = wandb.config
    config.max_len = 11
    config.buckets = 20

    # Cache pre-processed data
    if len(glob.glob("*.npy")) == 0:
        save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

    labels = ["bed", "happy", "cat"]


    # # Loading train set and test set
    X_train, X_test, y_train, y_test = get_train_test()

    # # Feature dimension
    channels = 1
    config.epochs = 50
    config.batch_size = 100

    num_classes = 3

    X_train = X_train.reshape(
        X_train.shape[0], config.buckets, config.max_len, channels)
    X_test = X_test.reshape(
        X_test.shape[0], config.buckets, config.max_len, channels)

    y_train_hot = tf.keras.utils.to_categorical(y_train)
    y_test_hot = tf.keras.utils.to_categorical(y_test)

    #model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.Flatten(input_shape=(config.buckets, config.max_len, channels)))
    #model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    inp = tf.keras.Input((config.buckets, config.max_len, channels))
    x = tf.keras.layers.Conv2D(wandb.config.conv1, kernel_size=(3,3), strides=1,activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(wandb.config.conv2, kernel_size=(3,3), strides=1,activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),strides=1,activation='relu')(x)
    x = tf.keras.layers.Conv2D(wandb.config.conv3, kernel_size=(1,1), strides=1,activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(wandb.config.dense1, activation="relu")(x)
    x = tf.keras.layers.Dense(wandb.config.dense2, activation="relu")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, x)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    config.total_params = model.count_params()
    print("Params: ", config.total_params)
    if config.total_params < 5000:

        model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(
            X_test, y_test_hot), callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])

sweep_config = {
    'method': 'random',
    'parameters': {
        'conv1': {
            'values': [8, 16, 32, 64]
        },
        'conv2': {
            'values': [32, 64, 96, 128]
        },
        'conv3': {
            'values': [6, 8, 10, 12, 14]
        },
        'dense1': {
            'values': [4, 8, 10, 12]
        },
        'dense2': {
            'values': [16, 32, 64, 96, 128, 256]
        }
    }
}

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train)
