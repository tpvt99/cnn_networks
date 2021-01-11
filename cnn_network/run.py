import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# Resnet
from cnn_network.resnet.models import resnet34, resnet50
from cnn_network.resnet.train_scripts import ResnetTrainable
# VGG

#

import sys
sys.path.append('../')
from logger.Logger import Logger

def run(config):
    #1. Initialize loggers
    logger = Logger(exp_name = config['exp_name'])

    #2. Data Loader
    if config['dataset'] == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        train_dataset = train_dataset.shuffle(buffer_size=100000).batch(batch_size=config['batch_size'])
        input_shape = train_images.shape[1:]
        output_size = 10
    else:
        raise ValueError(f"Please provide this dataset for {config['dataset']}")

    #3. Model Loader
    if config['exp_name'] == 'resnet34':
        model = resnet34(inputs = tf.keras.Input(shape = input_shape), output_size=output_size)
    else:
        raise ValueError(f"Please provide this cnn_network for {config['exp_name']}")
    print(model.summary())
    #4. Trainer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    trainer = ResnetTrainable(logger = logger, optimizer = optimizer,
                              model = model, train_dataset = train_dataset, test_dataset = test_dataset, epochs=config['epochs'])

    #5. Write the config.json files
    logger.save_config(config)
    logger.setup_tf_saver(model)

    #6. Training
    trainer.run()


if __name__ == "__main__":
    # 1. Argument Parser
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, required=True) # The cnn_network you want to run
    parser.add_argument('--dataset', type=str, required=True) # The dataset you want to run
    args = parser.parse_args()

    params = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'epochs': 10,
        'batch_size': 128,
        'learning_rate': 1e-3,
    }

    run(params)