from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, Dropout, Activation
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from math import ceil, floor, sqrt
from PIL import Image
import pandas as pd
import numpy as np
import os

# Global variables for training and testing
BATCH_SIZE = 32
EPOCHS = 25
INIT_LR = 1e-3
input_width, input_height = 224, 224
train_img, train_loc, train_label  = [], [], []
test_img, test_loc, test_label = [], [], []
label = ["paper", "rock", "scissors"]

# Function for getting the coordinates of the bounding box from list
def getBoxLoc(box_coord, width, height):
    xmin = int(box_coord['xmin'])
    ymin = int(box_coord['ymin'])
    xmax = int(box_coord['xmax'])
    ymax = int(box_coord['ymax'])
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > width:
        xmax = width
    if ymax > height:
        ymax = height
    loc = [floor(xmin*input_width/width),
           floor(ymin*input_height/height),
           floor(xmax*input_width/width),
           floor(ymax*input_height/height)]
    return loc

# Import image, location and label
data_dir = os.getcwd()+"/data"
# Training set
for i in range(3):
    path = data_dir+'/train/'+label[i]
    for file in os.listdir(path):
        fileType = file[len(file)-3:len(file)]
        if fileType != "png" and fileType != "jpg":
            continue
        else:
            fileName = file[0:len(file)-4]
        img = Image.open(path+"/"+file)
        x = img.width
        y = img.height
        img = np.array(img.resize((input_width, input_height)))
        train_img.append(img)
        json_str = open(path+"/outputs/"+fileName+".json").read()
        json_df = pd.read_json(json_str, orient = "record")
        train_loc.append(getBoxLoc(json_df.values[2][1][0]['bndbox'], x, y))
        train_label.append(i)
# Testing set
for i in range(3):
    path = data_dir+'/test/'+label[i]
    for file in os.listdir(path):
        fileType = file[len(file)-3:len(file)]
        if fileType != "png" or fileType != "jpg":
            continue
        else:
            fileName = file[0:len(file)-4]
        img = Image.open(path+"/"+file)
        w = img.width
        h = img.height
        img = np.array(img.resize((input_width, input_height)))
        test_img.append(img)
        json_str = open(path+"/outputs/"+fileName+".json").read()
        json_df = pd.read_json(json_str, orient = "record")
        test_loc.append(getBoxLoc(json_df.values[2][1][0]['bndbox'], w, h))
        test_label.append(i)

# Pose branch
def branchPose(inputs, numPose):
    x = MobileNet(weights = 'imagenet',include_top = False)(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(numPose)(x)
    x = Activation("softmax", name = "pose_output")(x)
    return x

# Location branch
def branchLocation(inputs):
    y =

# Build the final architecture
def buildNet():
    inputs = Input(shape = (input_width, input_height, 3))
    pose = branchPose(inputs, len(label))
    location = branchLocation(inputs)
    model = Model(inputs = inputs, outputs = [pose, location], name = "pphp2019")
    return model

# Initialize our network and loss function
model = buildNet()
losses = {
    "pose_output": "binary_crossentropy",
    "location_output": ""
}
lossWeights = {"pose_output": 1.0, "location_output": 1.0}

#Train our network
opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(optimizer = opt, loss = losses, lossWeights = lossWeights, metrics = ["accuracy"])
best_model_weights = './model_web.h5'
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
model.fit(x = train_img,
          y = {"pose_output": train_label, "location_output": train_loc},
          validation_data = (test_img, 
                             {"pose_output": test_label, "location_output": test_loc}),
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          steps_per_epoch = ceil(len(train_label)/BATCH_SIZE),
          validation_steps = ceil(len(test_label)/BATCH_SIZE),
          verbose = 1,
          callbacks = checkpoint)

# Test our network
model.load_weights(best_model_weights)
model_score = model.evaluate(test_img,
                             {"pose_output": test_label, "location_output": test_loc},
                             batch_size = BATCH_SIZE,
                             steps = ceil(len(test_label)/BATCH_SIZE),
                             verbose = 1)
print("Model Test Loss:", model_score[0])
print("Model Test Accuracy:", model_score[1])
