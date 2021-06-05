import os
import sys
import warnings

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images

from tensorflow import keras

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input
# from keras.layers.core import Dropout, Lambda
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from tensorflow.keras.layers import Activation, BatchNormalization
# from tensorflow.keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Lambda
# from keras import backend as K
# from keras.layers.merge import concatenate
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
# tf.keras.backend.clear_session()

# Set some parameters
BATCH_SIZE = 15  # the higher the better
IMG_WIDTH = 640  # for faster computing on kaggle
IMG_HEIGHT = 960  # for faster computing on kaggle
IMG_CHANNELS = 3
TRAIN_PATH = './train/'
TEST_PATH = './test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 12

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
np.random.seed(10)

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = cv2.imread(path + '/img/' + id_ + '.png')
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    X_train[n] = img

    mask_file = next(os.walk(path + '/mask/'))[2]
    mask = cv2.imread(path + '/mask/' + mask_file[0])
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.expand_dims(cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH)), axis=-1)
    Y_train[n] = mask / 255

    cv2.imshow('image', img)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    break

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = cv2.imread(path + '/img/' + id_ + '.png')
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    sizes_test.append([img.shape[0], img.shape[1]])
    X_test[n] = img

print('Done!')

from keras.preprocessing import image

# Creating the training Image and Mask generator
image_datagen = image.ImageDataGenerator(rescale=1./255)
# fill_mode='nearest' to try
# brightness_range=[0.4,1.5]
mask_datagen = image.ImageDataGenerator()

# Keep the same seed for image and mask generators so they fit together
percent = 0.9
image_datagen.fit(X_train[:int(X_train.shape[0] * percent)], augment=True, seed=seed)
mask_datagen.fit(Y_train[:int(Y_train.shape[0] * percent)], augment=True, seed=seed)

x = image_datagen.flow(X_train[:int(X_train.shape[0] * percent)], batch_size=BATCH_SIZE, shuffle=True, seed=seed)
y = mask_datagen.flow(Y_train[:int(Y_train.shape[0] * percent)], batch_size=BATCH_SIZE, shuffle=True, seed=seed)

# Creating the validation Image and Mask generator
image_datagen_val = image.ImageDataGenerator(rescale=1./255)
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(X_train[int(X_train.shape[0] * percent):], augment=True, seed=seed)
mask_datagen_val.fit(Y_train[int(Y_train.shape[0] * percent):], augment=True, seed=seed)

x_val = image_datagen_val.flow(X_train[int(X_train.shape[0] * percent):], batch_size=BATCH_SIZE, shuffle=True, seed=seed)
y_val = mask_datagen_val.flow(Y_train[int(Y_train.shape[0] * percent):], batch_size=BATCH_SIZE, shuffle=True, seed=seed)

# # Checking if the images fit
# imshow(x.next()[0].astype(np.uint8))
# print(np.unique(x.next()[0], return_counts=True))
# plt.show()
# imshow(np.squeeze(y.next()[0].astype(np.uint8)))
# plt.show()
# imshow(x_val.next()[0].astype(np.uint8))
# plt.show()
# imshow(np.squeeze(y_val.next()[0].astype(np.uint8)))
# plt.show()

# creating a training and validation generator that generate masks and images
train_generator = zip(x, y)
val_generator = zip(x_val, y_val)


def iou_metric_no_smooth(y_true, y_pred):
    pred_flat = tf.reshape(y_pred, [-1, IMG_HEIGHT * IMG_WIDTH])
    true_flat = tf.reshape(y_true, [-1, IMG_HEIGHT * IMG_WIDTH])

    smooth = 1.
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth

    iou_score = tf.reduce_mean(intersection / denominator)
    return iou_score


def iou_loss(y_true, y_pred):
    pred_flat = tf.reshape(y_pred, [-1, IMG_HEIGHT * IMG_WIDTH])
    true_flat = tf.reshape(y_true, [-1, IMG_HEIGHT * IMG_WIDTH])

    smooth = .001
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth

    iou_score = tf.reduce_mean(intersection / denominator)
    return 1 - iou_score


def Unet(img_height, img_width, img_channels, nclasses=1, filters=32, drop_out=0.1):
    inputs = Input((img_height, img_width, img_channels))
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)  # s daca normalizam
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


model = Unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[iou_metric_no_smooth])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-traffic-10.h5', verbose=1, save_best_only=True)
results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=150,
                              epochs=5, verbose=1,
                              callbacks=[earlystopper, checkpointer])
model.save('last_model10.h5')
