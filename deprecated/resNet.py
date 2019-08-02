from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense,Dropout,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import os
import math
import sys

# Import image data via keras API
# Modifty size. Add random flipping to recognize direction
data_dir = os.getcwd()+"/" + sys.argv[1]
batch_size = 32
augs_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip = True,
    validation_split = 0.5
)

train_gen = augs_gen.flow_from_directory(
    data_dir,
    target_size = (224,224),
    batch_size=batch_size,
    class_mode = 'categorical',
    shuffle=True,
    subset = 'training'
)

val_gen = augs_gen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset = 'validation'
)

model = Sequential()
model_base = ResNet50(weights='imagenet',include_top=False,input_shape=train_gen.image_shape,pooling='avg')
model.add(model_base)
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

best_model_weights = './model_resNet.h5'
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