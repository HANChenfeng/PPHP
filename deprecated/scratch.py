from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D, Lambda
from keras.optimizers import *
import numpy as np
import tensorflow as tf
import math
import os

# Data import
# Uses keras's API to achieve splitting, random flipping and vertical shift
data_dir = os.getcwd() + "/data"
augs_gen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    height_shift_range = .2,
    vertical_flip = True,
    validation_split = 0.3
)

train_gen = augs_gen.flow_from_directory(
    data_dir,
    target_size = (224,224),
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = True,
    subset = 'training'
)

val_gen = augs_gen.flow_from_directory(
    data_dir,
    target_size = (224,224),
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False,
    subset = 'validation'
)

# Implementation of the Canny edge detector
def canny(img):
    # Generating gaussian matrix from N(0,1)
    sigma1 = sigma2 = 1
    sum = 0
    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                                + (np.square(j - 3) / np.square(sigma2)))) / (
                                         2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]
    gaussian = gaussian / sum

    # Conversion from rpg to gray scale
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # Gaussian filter
    with tf.Session() as sess:
        gray = rgb2gray(img.eval())
    W, H = gray.shape
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # Find the intensity gradient
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

    # Non-maximum Suppression
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]
                # Greater Y gradient value
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # Same gradient sign
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # Different gradient sign
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # Greater X gradient value
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # Same gradient sign
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # Different gradient sign
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    # Hysteresis Thresholding
    W3, H3 = NMS.shape
    DT = np.zeros([W3, H3])
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1

    return DT

# Network architecture
model = Sequential()
# Convert image to only edge
model.add(Lambda(canny, input_shape=train_gen.image_shape))
# Convolution and pooling layer
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

opt = Adam()
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.fit_generator(
    train_gen,
    steps_per_epoch = math.ceil(train_gen.n/1),
    validation_data  = val_gen,
    validation_steps = math.ceil(val_gen.n/1),
    epochs = 10,
    verbose = 1,
)

model.save(os.getcwd() + "/model.h5")