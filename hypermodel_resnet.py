from kerastuner import HyperModel
from tensorflow import keras
from loguru import logger


class RESNETHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):

        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):

        n_feature_maps = 64
        input_layer = keras.layers.Input(shape=self.input_shape)
        # BLOCK 1
        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=8, padding="same"
        )(input_layer)

        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=1, padding="same"
        )(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation("relu")(output_block_1)

        # BLOCK 2
        conv_x = keras.layers.Conv1D(
            filters=hp.Choice(
                "num_filter_1",
                values=[2 * 32, 2 * 64],
                default=2 * 64,
            ),
            kernel_size=hp.Choice(
                "kernel_size_1",
                values=[8, 16],
                default=8,
            ),
            padding="same",
        )(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=hp.Choice(
                "num_filter_2",
                values=[2 * 32, 2 * 64],
                default=2 * 64,
            ),
            kernel_size=hp.Choice(
                "kernel_size_2",
                values=[5, 10],
                default=5,
            ),
            padding="same",
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2,
            kernel_size=3,
            padding="same",
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=1, padding="same"
        )(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation("relu")(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(
            filters=hp.Choice(
                "num_filter_4",
                values=[2 * 32, 2 * 64],
                default=2 * 64,
            ),
            kernel_size=hp.Choice(
                "kernel_size_4",
                values=[8, 16],
                default=8,
            ),
            padding="same",
        )(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=hp.Choice(
                "num_filter_5",
                values=[2 * 32, 2 * 64],
                default=2 * 64,
            ),
            kernel_size=hp.Choice(
                "kernel_size_5",
                values=[5, 10],
                default=5,
            ),
            padding="same",
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2,
            kernel_size=3,
            padding="same",
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation("relu")(output_block_3)

        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(self.num_classes, activation="sigmoid")(
            gap_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        logger.info(f"output_layer shape : {output_layer.shape}")
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model
