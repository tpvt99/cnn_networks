import tensorflow as tf
import math

class ResNext():
    def __init__(self, stage_depths: tuple, baseWidth: int, cardinality: int, outputSizes: int):
        '''
        Use Functional API to build ResNext
        :param stage_depths: depth of each stages, e.g., for ResneXt-50, this should be [3,4,6,3]
        :param baseWidth: bottle-neck width (d) of first stage
        :param cardinality: cardinality
        :param outputSizes: output size of final FC layer
        '''
        # For ResNext-50 use depths=[3,4,6,3]
        # For ResNext-101 use depths=[3,4,23,3]
        assert len(stage_depths) == 4, "ResNext follows Resnet depths so depths must have length of 4"

        self.stage_depths = stage_depths
        self.nFeatures = [64, 128, 256, 512]
        # For baseWidth and cardinality, please use pair to have same complexity as Resnet
        # C: 32 - bW 4
        # C: 8 - bW 14
        # C: 4 - bW 24
        # C: 2 - bW 40
        # C: 1 - bW 64
        self.baseWidth = baseWidth
        self.cardinality = cardinality
        self.outputSizes = outputSizes

        # To keep track the output channels after each conv2-3-4-5 block
        self.currentOutputChannels = None

    def split(self,  inputs: tf.Tensor, inputChannels: int, outputChannels: int, d: int, cardinality: int, strides:int):
        '''
        split function for each type A/B/C
        :return:
        '''
        raise NotImplementedError

    def build_layer(self, inputs: tf.Tensor, depth:int, features: int, strides: int):
        # First block
        d = math.floor(features * (self.baseWidth/64))
        X = self.split(inputs, inputChannels = self.currentOutputChannels, outputChannels=features*4,
                             d=d, cardinality=self.cardinality, strides=strides)
        # depth-1 Continuous blocks
        for _ in range(1, depth):
            X = self.split(X, inputChannels=features*4, outputChannels=features*4,
                                 d=d, cardinality=self.cardinality, strides=1)

        self.currentOutputChannels = features * 4 # Update this info to feed to next build_layer call
        return X

    def build_model(self):
        inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
        # Conv1
        X = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name='conv1_conv')(inputs)
        X = tf.keras.layers.BatchNormalization(name='conv1_bn')(X)
        X = tf.keras.layers.ReLU(name='conv1_relu')(X)

        # Conv2, pool layer
        X = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='conv2_pool')(X)
        self.currentOutputChannels = 64
        # Conv2 with convolution layers
        X = self.build_layer(X, depth=self.stage_depths[0], features=self.nFeatures[0], strides=1)
        # Conv3
        X = self.build_layer(X, depth=self.stage_depths[1], features=self.nFeatures[1], strides=2)
        # Conv4
        X = self.build_layer(X, depth=self.stage_depths[2], features=self.nFeatures[2], strides=2)
        # Conv5
        X = self.build_layer(X, depth=self.stage_depths[3], features=self.nFeatures[3], strides=2)

        # Average Pool
        X = tf.keras.layers.GlobalAveragePooling2D()(X)

        # Fully Connected layer
        X = tf.keras.layers.Dense(units=self.outputSizes)(X)
        X = tf.keras.layers.Softmax(axis=-1)(X)

        models = tf.keras.Model(inputs = inputs, outputs = X)

        return models

class ResNextTypeA(ResNext):
    def __init__(self, stage_depths: tuple, baseWidth: int, cardinality: int, outputSizes: int):
        super().__init__(stage_depths, baseWidth, cardinality, outputSizes)

    def split(self, inputs: tf.Tensor, inputChannels: int, outputChannels: int, d: int, cardinality: int, strides:int):
        concat = []
        for i in range(cardinality):
            X = tf.keras.layers.Conv2D(filters=d, kernel_size=1, padding='same', strides=strides)(inputs)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.ReLU()(X)

            X = tf.keras.layers.Conv2D(filters=d, kernel_size=3, padding='same', strides=1)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.ReLU()(X)

            X = tf.keras.layers.Conv2D(filters=outputChannels , kernel_size=1, padding='same', strides=1)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            concat.append(X)
        out = tf.keras.layers.Add()(concat)

        if strides > 1 or inputChannels != outputChannels:
            residuals = tf.keras.layers.Conv2D(filters=outputChannels, kernel_size=1, padding='same', strides=strides)(inputs)
            residuals = tf.keras.layers.BatchNormalization()(residuals)
        else:
            residuals = inputs

        out = tf.keras.layers.Add()([out, residuals])
        out = tf.keras.layers.ReLU()(out)
        return out


class ResNextTypeB(ResNext):
    def __init__(self, stage_depths: tuple, baseWidth: int, cardinality: int, outputSizes: int):
        super().__init__(stage_depths, baseWidth, cardinality, outputSizes)

    def split(self, inputs: tf.Tensor, inputChannels: int, outputChannels: int, d: int, cardinality: int, strides:int):
        concat = []
        for i in range(cardinality):
            X = tf.keras.layers.Conv2D(filters=d, kernel_size=1, strides=strides, padding='same')(inputs)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.ReLU()(X)

            X = tf.keras.layers.Conv2D(filters=d, kernel_size=3, strides=1, padding='same')(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.ReLU()(X)
            concat.append(X)

        X = tf.keras.layers.Concatenate(axis=-1)(concat)
        X = tf.keras.layers.Conv2D(filters=outputChannels, kernel_size=1, padding='same', strides=1)(X)
        X = tf.keras.layers.BatchNormalization()(X)

        if strides > 1 or inputChannels != outputChannels:
            residuals = tf.keras.layers.Conv2D(filters=outputChannels, kernel_size=1, strides=strides, padding='same')(inputs)
            residuals = tf.keras.layers.BatchNormalization()(residuals)
        else:
            residuals = inputs

        out = tf.keras.layers.Add()([X, residuals])
        out = tf.keras.layers.ReLU()(out)

        return out

class ResNextTypeC(ResNext):
    def __init__(self, stage_depths: tuple, baseWidth: int, cardinality: int, outputSizes: int):
        super().__init__(stage_depths, baseWidth, cardinality, outputSizes)

    def split(self, inputs: tf.Tensor, inputChannels: int, outputChannels: int, d: int, cardinality: int, strides: int):
        X = tf.keras.layers.Conv2D(filters=d*cardinality, kernel_size=1, strides=strides, padding='same')(inputs)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.ReLU()(X)

        X = tf.keras.layers.Conv2D(filters=d*cardinality, kernel_size=3, strides=1, padding='same', groups=cardinality)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.ReLU()(X)

        X = tf.keras.layers.Conv2D(filters=outputChannels, kernel_size=1, strides=1, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)

        if strides > 1 or inputChannels != outputChannels:
            residuals = tf.keras.layers.Conv2D(filters=outputChannels, kernel_size=1, strides=strides, padding='same')(
                inputs)
            residuals = tf.keras.layers.BatchNormalization()(residuals)
        else:
            residuals = inputs
        out = tf.keras.layers.Add()([X, residuals])
        out = tf.keras.layers.ReLU()(out)

        return out


models = ResNextTypeC(stage_depths=(3,4,6,3), baseWidth=4, cardinality=32, outputSizes=1000).build_model()
models.summary()
tf.keras.utils.plot_model(models, "my_first_model_with_shape_info.png", show_shapes=True)
