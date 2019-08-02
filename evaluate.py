from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
import keras
import os
import math
import time

# Import image data via keras API
# Modifty size. Add random flipping to recognize direction
data_dir = os.getcwd()+"/data"
batch_size = 32
augs_gen = ImageDataGenerator(
    rescale=1./255,
)

val_gen = augs_gen.flow_from_directory(
    data_dir + "/test",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Handle relu6 error
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                        'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    # Load the model and test
    scratch = load_model(os.getcwd()+"/model_scratch.h5")
    mobile = load_model(os.getcwd()+"/model_mobile.h5")
    res = load_model(os.getcwd()+"/model_resNet.h5")
    vgg = load_model(os.getcwd() + "/model_vgg.h5")
    t0 = time.time()
    y = scratch.evaluate_generator(val_gen, steps=math.ceil(val_gen.n/batch_size))
    t1 = time.time()
    print(y)
    print(t1-t0)
    t0 = time.time()
    y = mobile.evaluate_generator(val_gen, steps=math.ceil(val_gen.n/batch_size))
    t1 = time.time()
    print(y)
    print(t1-t0)
    t0 = time.time()
    y = res.evaluate_generator(val_gen, steps=math.ceil(val_gen.n/batch_size))
    t1 = time.time()
    print(y)
    print(t1-t0)
    t0 = time.time()
    y = vgg.evaluate_generator(val_gen, steps=math.ceil(val_gen.n/batch_size))
    t1 = time.time()
    print(y)
    print(t1-t0)
