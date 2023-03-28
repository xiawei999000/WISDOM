'''
use the CIFIR10 images to pretrain the intensity diagnostic network
author: xiav
'''

import matplotlib.pyplot as plt
import os
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
from tensorflow import keras

# basic settings
base_model_name = 'resnet_18_3'
img_type = 'CIFIR10'
initial_learning_rate = 0.001
batch_size = 128
num_classes = 10
epochs = 70
metric_val = 'acc'

# enable the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# load data set
(train_images_cifar, train_labels_cifar), (test_images_cifar, test_labels_cifar) = cifar10.load_data()
train_images_cifar = train_images_cifar.reshape((50000, 32, 32, 3))
train_images_cifar = train_images_cifar.astype('float32') / 255

test_images_cifar = test_images_cifar.reshape((10000, 32, 32, 3))
test_images_cifar = test_images_cifar.astype('float32') / 255

train_labels_cifar = to_categorical(train_labels_cifar)
test_labels_cifar = to_categorical(test_labels_cifar)

X_train = train_images_cifar[0:45000]
X_val = train_images_cifar[45000:]
y_train = train_labels_cifar[0:45000]
y_val = train_labels_cifar[45000:]
X_test = test_images_cifar
y_test = test_labels_cifar
print("X_train.shape:", X_train.shape)
print("X_val.shape:", X_val.shape)
print("X_test.shape:", X_test.shape)

import random
for i in range(0, 6):
    plt.subplot(1, 6, i+1)
    test_im = X_train[random.randint(0, 600)]
    plt.imshow(test_im.reshape(32, 32, 3), cmap='viridis', interpolation='none')
    plt.tight_layout()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255


# define model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, BatchNormalization, Conv2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import add, Flatten

def Conv2d_BN(x, output_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(output_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Block(inpt, output_filter, kernel_size, strides=(1, 1), padding='same', with_conv_shortcut=False):
    x = Conv2d_BN(inpt, output_filter=output_filter, kernel_size=kernel_size, strides=strides, padding=padding)
    x = Conv2d_BN(x, output_filter=output_filter, kernel_size=kernel_size, padding=padding)
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, output_filter=output_filter, kernel_size=kernel_size, strides=strides,
                             padding=padding)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

input_tensor = Input(shape=(32, 32, 3))

# conv1
x = Conv2d_BN(input_tensor, 64, (3, 3), (2, 2), padding='same')

# conv2
x = Block(x, output_filter=64, kernel_size=(3, 3))
x = Block(x, output_filter=64, kernel_size=(3, 3))

# conv3
x = Block(x, output_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
x = Block(x, output_filter=128, kernel_size=(3, 3))

# # conv4
# x = Block(x, output_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
# x = Block(x, output_filter=256, kernel_size=(3, 3))

# # conv5
# x = Block(x, output_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
# x = Block(x, output_filter=512, kernel_size=(3, 3))
#
# x = AveragePooling2D(pool_size=(2, 2))(x)
# x = Flatten()(x)

x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=x)
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=initial_learning_rate),
              metrics=[metric_val])

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)


# Define callbacks.
model_version_save_path = './models/' + base_model_name
if not os.path.exists(model_version_save_path):
    os.makedirs(model_version_save_path)

model_save_path = model_version_save_path + '/' + img_type + '_' + base_model_name + '_lr-' + str(initial_learning_rate) \
                  + '_batchSize-' + str(batch_size)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
output_model_file = model_save_path + '/epoch-{epoch:02d}_val_' + metric_val + '-{val_' + metric_val + ':.2f}.hdf5'

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    output_model_file, monitor='val_' + metric_val, save_best_only=False, save_weights_only=False, mode='auto', period=1
)
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_' + metric_val, factor=0.5, patience=10, mode='auto')

# call back functions
callbacks = [checkpoint_cb, lr_reducer]  # , lr_scheduler, EarlyStop

# Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(datagen.flow(X_train, y_train,
                                           batch_size=batch_size),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              epochs=epochs,
                              validation_data=(X_val, y_val),
                              callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


import matplotlib.pyplot as plt
# matplotlib inline
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
