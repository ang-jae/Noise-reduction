import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add, UpSampling2D
import tensorflow.python.keras

import os
import matplotlib.pyplot as plt
import numpy as np
import math

train_datagen = ImageDataGenerator(rescale=1. / 255.)
validation_datagen = ImageDataGenerator(rescale=1. / 255.)

train_dir = os.path.join('C:/Users/USER/dataset/noise/')
validation_dir = os.path.join('C:/Users/USER/dataset/train/')

# print(train_dir, validation_dir)

# print(os.listdir(train_dir), os.listdir(validation_dir))

input_shape = (480, 640, 1)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=input_shape[:2],
                                                    color_mode='grayscale')
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=16,
                                                              target_size=input_shape[:2], color_mode='grayscale')

print(len(train_generator), len(validation_generator))
# number of classes
K = 4

input_tensor = Input(shape=(480, 640, 1), dtype='float32', name='input')


def conv1_layer(x):
    shortcut = x

    x = Conv2D(8, (3, 3), strides=(1, 1), padding = 'SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, (3, 3), strides=(1, 1), padding = 'SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), padding = 'SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding = 'SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print(x.shape)
    print(shortcut.shape)

    tf.reshape(shortcut, (480, 640, 128), name=None)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def conv_layers(x):
    shortcut = x
    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(16, (3, 3)  # 16 is number of filters and (3, 3) is the size of the filter.
                     , padding='same', input_shape=(480, 640, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # 2nd convolution layer
    model.add(Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # here compressed version

    # 3rd convolution layer
    model.add(Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    # 4rd convolution layer
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(Activation('sigmoid'))

    tf.reshape(shortcut, (480, 640, 128), name=None)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    model.summary()

    return x

def get_mse(original, x, height, width):
    total = 0.0
    for i in range(height):
        for j in range(width):
            for k in range(3):
                total += (original[i][j][k] - x[i][j][k])**2 + (original[i][j][k] - x[i][j][k])**2 + (original[i][j][k] - x[i][j][k])**2
    return total / (width * height * 3)

########################################################################################3
x = conv_layers(input_tensor)

output_tensor = (480, 640, 128)

resnet5 = Model(input_tensor, output_tensor, name='ResNet5')

#########################
######reducing error######
from tensorflow.python.keras.backend import eager_learning_phase_scope

f = K.function([x.model.layers[0].input], [x.model.output])

# Run the function for the number of mc_samples with learning_phase enabled
with eager_learning_phase_scope(value=1):  # 0=test, 1=train
    Yt_hat = np.array([f((x))[0] for _ in range(train_datagen)])

########################

resnet5.compile(optimizer = tf.keras.optimizers.Adam(0.01),
             loss = 'categorical_crossentropy',
             metrics = [tf.keras.metrics.CategoricalAccuracy()])

num_epochs = 10
batch_size = 16

print(len(validation_generator), 11807/batch_size)

hist = resnet5.fit(train_generator, train_generator, batch_size = batch_size, shuffle = True, epochs = num_epochs)

resnet5.fit_generator(
        train_generator,
        steps_per_epoch = len(train_generator),
        epochs = num_epochs,
        validation_data = validation_generator,
        validation_steps = len(validation_generator)
)