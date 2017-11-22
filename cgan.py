import numpy as np
import sklearn as skl
from sklearn.preprocessing import OneHotEncoder
import scipy.io as scio
import scipy
import skimage as sk
import skimage.io as skio
from skimage import color
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import h5py
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Flatten, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Input
from PIL import Image
import math

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# svhn_train = scio.loadmat('svhn_train.mat')
# svhn_test = scio.loadmat('svhn_test.mat')
#
# svhn_train_X = svhn_train['X']
# svhn_train_y = svhn_train['y']
# svhn_test_X = svhn_test['X']
# svhn_test_y = svhn_test['y']
# encoder = OneHotEncoder()
# svhn_train_y = encoder.fit_transform(svhn_train_y).todense()
# svhn_test_y = encoder.fit_transform(svhn_test_y).todense()

def resize_images(imgs):
    return np.array([sk.img_as_float(scipy.misc.imresize(img.reshape(28, 28), (32, 32))) for img in imgs])

def normal_resize(imgs):
    x = np.array([np.reshape(a, (28, 28)) for a in imgs])
    return x[:,:,:,None]
# svhn_train_X = np.array([sk.img_as_float(scipy.misc.imresize(svhn_train_X[:, :, :, i], (32, 32, 3))) for i in range(svhn_train_X.shape[-1])])
# svhn_test_X = np.array([sk.img_as_float(scipy.misc.imresize(svhn_test_X[:, :, :, i], (32, 32, 3))) for i in range(svhn_test_X.shape[-1])])

# mnist_train_X = normal_resize(mnist_train_X)
# mnist_test_X = normal_resize(mnist_test_X)
# print(mnist_train_X.shape)
# stacked_mnist_train_X = np.stack((mnist_train_X, mnist_train_X, mnist_train_X), axis=3).reshape(mnist_train_X.shape[0], 32, 32, 3)
# stacked_mnist_test_X = np.stack((mnist_test_X, mnist_test_X, mnist_test_X), axis=3).reshape(mnist_test_X.shape[0], 32, 32, 3)

def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator():
    # return Sequential([
    #     Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 1), kernel_initializer='glorot_normal'),
    #     BatchNormalization(),
    #     LeakyReLU(alpha=0.3),
    #     MaxPooling2D(),
    #     Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal'),
    #     BatchNormalization(),
    #     LeakyReLU(alpha=0.3),
    #     MaxPooling2D(),
    #     Conv2D(1, (4, 4), padding='valid', kernel_initializer='glorot_normal'),
    #     BatchNormalization(),
    #     LeakyReLU(alpha=0.3),
    #     MaxPooling2D(),
    #     Flatten(),
    #     Dense(1024),
    #     Activation('relu'),
    #     Dense(1),
    #     Activation('sigmoid'),
    # ])
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_with_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

BATCH_SIZE = 128
N_EPOCHS = 100
def train(epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist_input = mnist.train.images
    mnist_input = np.array([np.reshape(a, (28, 28)) for a in mnist_input])
    mnist_input = mnist_input[:, :, :, None]
    X_test = mnist.test.images
    y_train = mnist.train.labels
    y_test = mnist.test.labels
    g = generator()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d = discriminator()
    g_d = generator_with_discriminator(g, d)
    g_d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_d.compile(loss='binary_crossentropy', optimizer=g_d_optim)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d.trainable = True
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for e in range(batch_size * epochs):
        print('Epoch:', e)
        for i in range(int(mnist_input.shape[0] / batch_size)):
            # Create batches
            batch_r_X = np.random.uniform(-1, 1, size=(batch_size, 100))
            batch_mnist_X = mnist_input[i*batch_size:(i+1)*batch_size]
            # batch_mnist_X = mnist_input[np.random.randint(mnist_input.shape[0], size=batch_size)]
            # Generate MNIST examples for SVHN for training
            g_train_data = g.predict(batch_r_X, verbose=0)
            # Label as real or fake
            real_labels, fake_labels = np.ones(batch_size), np.zeros(batch_size)
            train_data, train_labels = np.vstack((batch_mnist_X, g_train_data)), np.hstack((real_labels, fake_labels))
            # Train discriminator on this real/fake data
            d_loss = d.train_on_batch(train_data, train_labels)
            # Train generator based on the outputs of the discriminator
            d.trainable = False
            # should we train on new noise here?
            g_loss = g_d.train_on_batch(batch_r_X, np.ones(batch_r_X.shape[0]))
            d.trainable = True
            if i % 100 == 0:
                print('Iteration {}'.format(i))
                print('D Loss:', d_loss)
                print('G Loss:', g_loss)
                g.save_weights('G.h5')
                d.save_weights('D.h5')
        g_images = g.predict(np.random.normal(size=(10, 100)))
        image = combine_images(g_images)
        Image.fromarray(image.astype(np.uint8)).save(
                "epoch_"+str(e)+".png")
    return g, d

train()

def test_pred(g, D):
    g_images = g.predict(np.random.normal(size=(10, 100)))
    for i in range(10):
        plt.figure()
        plt.figure()
        plt.imshow(g_images[i][:, :, 0], cmap='gray')
    return D.predict(g_images)
