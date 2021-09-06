import numpy as np
import os
import sys
import tensorflow as tf
import cv2 as cv
from mss import mss
from PIL import Image

'''
def predictImage(filename, categories):
    image = tf.keras.preprocessing.image.load_img(
    filename, target_size=(212, 212))

    iarray = tf.keras.preprocessing.image.img_to_array(image)
    iarray = tf.expand_dims(iarray, 0)

    predict = model.predict(iarray)
    score = tf.nn.softmax(predict[0])

    print(
        "{} - confidence = {:.2f}"
        .format(categories[np.argmax(score)], 100 * np.max(score))
    )
'''
    
def predictLocalImage(model, image, categories):
    iarray = tf.keras.preprocessing.image.img_to_array(image)
    iarray = tf.expand_dims(iarray, 0)

    predict = model.predict(iarray)
    score = tf.nn.softmax(predict[0])
    
    return (categories[np.argmax(score)], 100 * np.max(score))

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 

print("TensorFlow version: {}".format(tf.__version__))

c1 = tf.keras.models.load_model("models/class1")

c1dict = {
    0: "smooth",
    1: "features or disc",
    2: "star or artifact",
}

while True:
    screenFrame = mss().grab({'left': 640, 'top': 480, 'width': 212, 'height': 212})
    
    frame = Image.frombytes(
        'RGB', 
        (screenFrame.width, screenFrame.height), 
        screenFrame.rgb,
    )
    
    cv.imshow('frame', np.array(frame))
    prediction = predictLocalImage(c1, frame, c1dict)
    print("{} - confidence = {:.2f}".format(prediction[0], prediction[1]))
    
    if cv.waitKey(1) == ord('q'):
        break
    
cv.destroyAllWindows()