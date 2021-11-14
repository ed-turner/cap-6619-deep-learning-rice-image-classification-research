import tensorflow as tf


def build_fcn_model(hp):
    """

    :param hp:
    :return:
    """

    num_layers = hp.Int("num_hid_layers", 0, 10)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28))
        ]
    )

    model.add(tf.keras.layers.Flatten())

    for i in range(num_layers):
        model.add(
            tf.keras.layers.Dense(hp.Int(f"hidden-layer-{i + 1}", 2, 128))
        )
        model.add(tf.keras.layers.Dropout(hp.Float(f"dropout-{i + 1}", 0., 0.5)))

    model.add(
        tf.keras.layers.Dense(1)
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=hp.Float("learning_rate", 1e-6, 1e-1, sampling="log")),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.AUC(from_logits=True, curve="PR")
        ]
                  )

    return model


def build_cnn_model(hp):
    """

    :param hp:
    :return:
    """

    num_layers = hp.Int("num_hid_layers", 0, 10)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28))
        ]
    )

    for i in range(num_layers):
        stride = hp.Int(f"cnn-layer-{i + 1}-stride", 1, 5)
        model.add(
            tf.keras.layers.Conv2D(
                hp.Int(f"cnn-layer-{i + 1}-filter", 2, 128),
                hp.Int(f"cnn-layer-{i + 1}-kernel", 2, 128),
                stride=(stride, stride),
                padding='same',
                activation="elu"
            )
        )
        model.add(tf.keras.layers.Dropout(hp.Float(f"dropout-{i + 1}", 0., 0.5)))

    model.add(tf.keras.layers.Flatten())

    model.add(
        tf.keras.layers.Dense(1)
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=hp.Float("learning_rate", 1e-6, 1e-1, sampling="log")),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.AUC(from_logits=True, curve="PR")
        ]
                  )

    return model
