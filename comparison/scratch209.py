from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import os
import math
import sys

# Import image data via keras API
# Modifty size. Add random flipping to recognize direction
data_dir = os.getcwd()+"/data"
batch_size = 32
augs_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip = True,
)

train_gen = augs_gen.flow_from_directory(
    data_dir + "/train",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode = 'categorical',
    shuffle=True
)

val_gen = augs_gen.flow_from_directory(
    data_dir + "/test",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu', input_shape=train_gen.image_shape))
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

best_model_weights = './model_scratch.h5'
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=40,
    verbose=1,
    mode='auto',
    cooldown=1
)

callbacks = [checkpoint, reduce]

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch = math.ceil(train_gen.n/batch_size),
    validation_data  = val_gen,
    validation_steps = math.ceil(val_gen.n/batch_size),
    epochs=25,
    verbose=1,
    callbacks=callbacks
)

model.load_weights(best_model_weights)
model_score = model.evaluate_generator(val_gen, steps=math.ceil(val_gen.n/batch_size))
print("Model Test Loss:", model_score[0])
print("Model Test Accuracy:", model_score[1])