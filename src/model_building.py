from tensorflow.keras import layers, Model
import tensorflow as tf
from typing import Tuple

def residual_block(x: tf.Tensor, filters: int, downsample: bool = False) -> tf.Tensor:
    """
    Constructs a ResNet-style residual block.
    
    :param x: The input tensor to the block.
    :param filters: The number of filters for the convolutional layers.
    :param downsample: If True, applies downsampling to the input.
    :return: The output tensor of the residual block.
    """
    identity = x
    
    # Check if a 1x1 convolution is needed for downsampling or filter count change
    if downsample or identity.shape[-1] != filters:
        identity = layers.Conv2D(filters, kernel_size=1, 
                                 strides=2 if downsample else 1, 
                                 padding="same")(identity)
        identity = layers.BatchNormalization()(identity)

    # First convolutional of the block 
    x = layers.Conv2D(filters, kernel_size=3, 
                      strides=2 if downsample else 1, 
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second convolutional of the block
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Add the identity (skip connection) to the output of the block
    x = layers.Add()([x, identity])
    x = layers.ReLU()(x)
    
    return x

def ResNet18(input_shape: Tuple[int, int, int] = (64, 64, 3), num_classes: int = 10) -> Model:
    """
    Creates a ResNet-18 model for image classification.
    
    :param input_shape: The shape of the input images as a tuple (height, width, channels).
    :param num_classes: The number of output classes for the final dense layer.
    :return: A Keras Model instance representing ResNet-18.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution and max pooling layers
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)
    
    # fIBnal layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs, name="ResNet-18")
    return model