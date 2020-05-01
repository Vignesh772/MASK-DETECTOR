import cv2

import tensorflow as tf

from keras.preprocessing import image


import tensorflow.keras as k
import cv2
import numpy as np
import time

model = tf.keras.models.load_model('static\\model2.h5')
face_cascade=cv2.CascadeClassifier("static\\haarcascade_frontalface_alt2.xml")
ds_factor=0.6

class VideoCamera(object):
    mask=0

    def __init__(self):

        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    def mask_status(self):
        return(self.mask)

    def get_frame(self):
        success, pic = self.video.read()
        pic=cv2.resize(pic,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        #print(pic.shape)

        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        img=pic.copy()
        for (x,y,w,h) in face_rects:
            img=img[y-10:y+h+10,x-10:x+w+10]
            break
        #cv2.imshow('frame',img)
        #k=cv2.waitKey(5)&0xFF
        cv2.imwrite('static\\img.png',img)







        path="static\\img.png"

        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes)


        if (classes[0]>0.5):
            col=(0,0,255)
            white=cv2.imread('static\\no_mask_text1.png')
            white=cv2.resize(white,(pic.shape[1],100))
            for (x,y,w,h) in face_rects:
                cv2.rectangle(pic,(x-10,y-10),(x+w+10,y+h+10),col,2)
                break
            self.mask=0
            print(" no mask")


        else:
            col=(0,255,0)
            white=cv2.imread('static\\mask_text1.png')
            white=cv2.resize(white,(pic.shape[1],100))
            self.mask=1

            print("maskkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")



        i=np.row_stack((pic,white))
        ret, jpeg = cv2.imencode('.jpg', i)
        return (jpeg.tobytes())
