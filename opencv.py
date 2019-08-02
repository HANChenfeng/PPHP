from keras.models import load_model
import numpy as np
import os
import cv2


model = load_model(os.getcwd() + '/model_mobile.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('Frame is none.')
        break
    cv2.imshow('frame', frame)
    frame = cv2.resize(frame, (224, 224))/255
    img = [frame,]
    img = np.array(img)
    y = model.predict_classes(img)
    if y==0:
        print("paper")
    elif y==1:
        print("rock")
    else:
        print("scissor")
    if cv2.waitKey(250) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()