import tensorflow as tf


def basic_blocks(inputs: tf.Tensor, filters: int, kernel_size: int, strides: int, name: str) -> tf.Tensor:
    '''
    Each block of resnet 34
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :return:
    '''
    X = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding='same', name=f'{name}_conv1')(inputs)
    X = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(X)
    X = tf.keras.layers.ReLU(name=f'{name}_relu1')(X)

    X = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=1, padding='same', name=f'{name}_conv2')(X)
    X = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(X)

    if strides > 1:
        residuals = tf.keras.layers.Conv2D(filters = filters, kernel_size=1, padding='valid', strides=strides, name=f'{name}_res_conv')(inputs)
        residuals = tf.keras.layers.BatchNormalization(name=f'{name}_res_bn')(residuals)
    else:
        residuals = inputs

    X = tf.keras.layers.Add( name=f'{name}_add')([residuals, X])
    X = tf.keras.layers.ReLU( name=f'{name}_relu2')(X)

    return X

def bottleneck_blocks(inputs: tf.Tensor, filters: tuple, kernel_size: int, strides: int, name: str) -> tf.Tensor:
    F1, F2, F3 = filters
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size=1, strides = strides, padding='same', name = f'{name}_conv1')(inputs)
    X = tf.keras.layers.BatchNormalization(name = f'{name}_bn1')(X)
    X = tf.keras.layers.ReLU(name = f'{name}_relu1')(X)

    X = tf.keras.layers.Conv2D(filters = F2, kernel_size=kernel_size, strides = 1, padding='same', name = f'{name}_conv2')(X)
    X = tf.keras.layers.BatchNormalization(name = f'{name}_bn2')(X)
    X = tf.keras.layers.ReLU(name = f'{name}_relu2')(X)

    X = tf.keras.layers.Conv2D(filters = F3, kernel_size=1, strides = 1, padding='same', name = f'{name}_conv3')(X)
    X = tf.keras.layers.BatchNormalization(name = f'{name}_bn3')(X)

    # We add the shape comparison to cover the dim mismatch in first block of conv2_x
    if strides > 1 or inputs.shape[-1] != X.shape[-1]:
        residuals = tf.keras.layers.Conv2D(filters = F3, kernel_size=1, strides = strides, padding='same', name = f'{name}_res_conv')(inputs)
        residuals = tf.keras.layers.BatchNormalization(name = f'{name}_res_bn')(residuals)
    else:
        residuals = inputs

    X = tf.keras.layers.Add(name = f'{name}_add')([residuals, X])
    X = tf.keras.layers.ReLU(name = f'{name}_relu3')(X)

    return X

def resnet34(inputs: tf.keras.Input = tf.keras.Input(shape=(224, 224, 3), name='input'), output_size: int = 1000):
    # Input
    if inputs is None:
        inputs = tf.keras.Input(shape=(224, 224, 3), name='input')

    # Conv1
    X = tf.keras.layers.Conv2D(filters = 64, kernel_size = 7, strides = 2, padding='same', name='conv1_conv')(inputs)
    X = tf.keras.layers.BatchNormalization(name='conv1_bn')(X)
    X = tf.keras.layers.ReLU(name='conv1_relu')(X)

    # Conv2, pool layer
    X = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name = 'conv2_pool')(X)
    # Conv2, block 1-2-3
    X = basic_blocks(X, filters=64, kernel_size=3, strides=1, name = 'conv2_block1')
    X = basic_blocks(X, filters=64, kernel_size=3, strides=1, name = 'conv2_block2')
    X = basic_blocks(X, filters=64, kernel_size=3, strides=1, name = 'conv2_block3')

    # Conv3, block 1-2-3-4
    X = basic_blocks(X, filters=128, kernel_size=3, strides=2, name = 'conv3_block1')
    X = basic_blocks(X, filters=128, kernel_size=3, strides=1, name = 'conv3_block2')
    X = basic_blocks(X, filters=128, kernel_size=3, strides=1, name = 'conv3_block3')
    X = basic_blocks(X, filters=128, kernel_size=3, strides=1, name = 'conv3_block4')

    # Conv4, block 1-2-3-4-5-6
    X = basic_blocks(X, filters=256, kernel_size=3, strides=2, name = 'conv4_block1')
    X = basic_blocks(X, filters=256, kernel_size=3, strides=1, name = 'conv4_block2')
    X = basic_blocks(X, filters=256, kernel_size=3, strides=1, name = 'conv4_block3')
    X = basic_blocks(X, filters=256, kernel_size=3, strides=1, name = 'conv4_block4')
    X = basic_blocks(X, filters=256, kernel_size=3, strides=1, name = 'conv4_block5')
    X = basic_blocks(X, filters=256, kernel_size=3, strides=1, name = 'conv4_block6')

    # Conv5, block 1-2-3
    X = basic_blocks(X, filters=512, kernel_size=3, strides=2, name = 'conv5_block1')
    X = basic_blocks(X, filters=512, kernel_size=3, strides=1, name = 'conv5_block2')
    X = basic_blocks(X, filters=512, kernel_size=3, strides=1, name = 'conv5_block3')

    # Average Pool
    X = tf.keras.layers.GlobalAveragePooling2D()(X)

    # Fully Connected layer
    X = tf.keras.layers.Dense(units = output_size)(X)
    X = tf.keras.layers.Softmax(axis=-1)(X)

    models = tf.keras.Model(inputs=inputs, outputs=X)
    return models

def resnet18():
    'You can depends on resnet34  to implement resnet18'
    raise NotImplementedError

def resnet50(inputs: tf.keras.Input = tf.keras.Input(shape=(224, 224, 3), name='input'), output_size: int = 1000):
    # Input
    if inputs is None:
        inputs = tf.keras.Input(shape=(224, 224, 3), name='input')

    # Conv1
    X = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name='conv1_conv')(inputs)
    X = tf.keras.layers.BatchNormalization(name='conv1_bn')(X)
    X = tf.keras.layers.ReLU(name='conv1_relu')(X)

    # Conv2, pool layer
    X = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='conv2_pool')(X)
    # Conv2, block 1-2-3
    X = bottleneck_blocks(X, filters=(64, 64, 256), kernel_size=3, strides=1, name = 'conv2_block1')
    X = bottleneck_blocks(X, filters=(64, 64, 256), kernel_size=3, strides=1, name = 'conv2_block2')
    X = bottleneck_blocks(X, filters=(64, 64, 256), kernel_size=3, strides=1, name = 'conv2_block3')

    #Conv3, block 1-2-3-4
    X = bottleneck_blocks(X, filters=(128, 128, 512), kernel_size=3, strides=2, name='conv3_block1')
    X = bottleneck_blocks(X, filters=(128, 128, 512), kernel_size=3, strides=1, name='conv3_block2')
    X = bottleneck_blocks(X, filters=(128, 128, 512), kernel_size=3, strides=1, name='conv3_block3')
    X = bottleneck_blocks(X, filters=(128, 128, 512), kernel_size=3, strides=1, name='conv3_block4')

    # Conv4, block 1-2-3-4-5-6
    X = bottleneck_blocks(X, filters=(256, 256, 1024), kernel_size=3, strides=2, name='conv4_block1')
    X = bottleneck_blocks(X, filters=(256, 256, 1024), kernel_size=3, strides=1, name='conv4_block2')
    X = bottleneck_blocks(X, filters=(256, 256, 1024), kernel_size=3, strides=1, name='conv4_block3')
    X = bottleneck_blocks(X, filters=(256, 256, 1024), kernel_size=3, strides=1, name='conv4_block4')
    X = bottleneck_blocks(X, filters=(256, 256, 1024), kernel_size=3, strides=1, name='conv4_block5')
    X = bottleneck_blocks(X, filters=(256, 256, 1024), kernel_size=3, strides=1, name='conv4_block6')

    #Conv5, block 1-2-3
    X = bottleneck_blocks(X, filters=(512, 512, 2048), kernel_size=3, strides=2, name='conv5_block1')
    X = bottleneck_blocks(X, filters=(512, 512, 2048), kernel_size=3, strides=1, name='conv5_block2')
    X = bottleneck_blocks(X, filters=(512, 512, 2048), kernel_size=3, strides=1, name='conv5_block3')

    # Average Pool
    X = tf.keras.layers.GlobalAveragePooling2D()(X)

    # Fully Connected layer
    X = tf.keras.layers.Dense(units=output_size)(X)
    X = tf.keras.layers.Softmax(axis=-1)(X)

    models = tf.keras.Model(inputs = inputs, outputs = X)

    return models

def resnet101():
    'You can depend on resnet50 to implement resnet101'
    raise NotImplementedError

def resnet152():
    'You can depend on resnet50 to implement resnet152'
    raise NotImplementedError
