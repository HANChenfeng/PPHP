from keras.models import load_model
from PIL import Image
import numpy as np
import os

m = load_model(os.getcwd()+"/model_mobile.h5")
f = os.listdir(os.getcwd()+"/selfdata/rock")[0]
img = Image.open(os.getcwd()+"/selfdata/rock/"+f).resize((224,224))
img3d = np.array(img)
img4d = np.array([img3d,])
m.predict_classes(img4d)