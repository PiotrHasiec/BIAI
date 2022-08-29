
# Implementacja obsługi ładowania i predykcji modelu
from keras import models
from keras.models import Model
import os
from keras.models import model_from_json
from keras.models import load_model

import cv2
import numpy as np
IMG_HEIGHT = 250
IMG_WIDTH = 250
def create_dataset(img_folder,max_count = 640):
    i = 0
    img_data_array=[]
    class_name=[]
    a = []
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir)):
            if not (file.endswith('.jpg') or file.endswith('.jpeg')or file.endswith('.png')):
                continue
            image_path= os.path.join(img_folder, dir,  file)
            image= cv2.imread( image_path,cv2.COLOR_BGR2GRAY)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image = image.astype('float32')
            image = (image/255)
            
            class_name.append(image)
            a.append([image,image])
            i +=1
            if i>max_count:
                np.random.seed(0)
                img_data_array = np.random.permutation(img_data_array)
                return ( np.array(class_name), np.array(class_name)), ( np.array(img_data_array), np.array(img_data_array))

           

    return ( np.array(class_name), np.array(class_name)), ( np.array(img_data_array), np.array(img_data_array))

from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QPixmap

def qimage_to_array(image):
    """
    Funkcja konwertująca obiekt QImage do numpy array
    """
    image = image.convertToFormat(QImage.Format_Grayscale8)
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    numpy_array = np.array(ptr).reshape(image.height(), image.width(), 1)
    return numpy_array
    
def get_values(model,dataset = ".\\lfw_funneled"):
   
    (train_images, train_labels), (test_images, test_labels) = create_dataset(dataset)   
    train_images = train_images.reshape((len(train_images), IMG_HEIGHT, IMG_HEIGHT, 3))
   
    enc = Model(inputs=model.input,outputs=model.get_layer('enc').output)
    x_enc = enc.predict(train_images, batch_size=64)

    x_mean = np.mean(x_enc, axis=0)

    x_cov = np.cov((x_enc - x_mean).T)
    evals, evecs = np.linalg.eig(x_cov)
    y_stdx_stds = np.std(x_enc, axis=0)
    c = np.dot(evecs, np.diag(np.sqrt(evals)))
    return [x_mean,y_stdx_stds,c]

def predict(input, model):

    
    prediction = model.predict(np.reshape(input,(-1,5*32) ))[0]*255
    prediction = np.array(prediction,dtype="uint8")
    img = np.reshape(prediction,(250,250,3)) 
    return img


def get_model(name ="model8" ):

    dec = models.Sequential()

    with open(name+".json", 'r') as plik:
        loaded_model_json = plik.read()
            
    model = model_from_json(loaded_model_json)
    model.load_weights(name+".h5")
    dec.add(model.get_layer('dec'))
    dec.add(model.get_layer('dec0'))
    dec.add(model.get_layer('dec2'))
    dec.add(model.get_layer('dec3'))
       
    dec.add(model.get_layer('dec4'))
    dec.compile(loss='mse')

    return model,dec  