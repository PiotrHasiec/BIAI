import os
import cv2
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# Ładowanie potrzebych modułów
# MNIST - zbiór obrazów z odręcznie pisanymi cyframi od 0 do 9
# Sequential- model sekwencyjny sieci neuronowej
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model, Sequential, load_model, model_from_json
from keras import layers
from keras import models
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical,plot_model
from matplotlib import pyplot as plt
import numpy as np
IMG_FOLDER = ".\\lfw_funneled"
IMG_HEIGHT = 250
IMG_WIDTH = 250
BATCH_SIZE = 16
NEW = True
# Wczytywanie danych
def create_dataset(img_folder, percent,max_count = 700):
    i = 0
    img_data_array=[]
    class_name=[]
    a = []
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir)):
            if not (file.endswith('.jpg') or file.endswith('.jpeg')):
                continue
            if i <0:
                i+=1
                continue
            image_path= os.path.join(img_folder, dir,  file)
            image= cv2.imread( image_path)
            image = image.astype('float32')
            image = (image/255)
            img_data_array.append([image])
            i +=1
            if i>max_count:
                up = int(len(img_data_array)*(percent))
                np.random.seed(0)
                img_data_array = np.random.permutation(img_data_array)
                return ( np.array(img_data_array[:up]), np.array(img_data_array[:up])), ( np.array(img_data_array[up:]), np.array(img_data_array[up:]))
            break

    up = int(len(img_data_array)*(percent))
    np.random.seed(0)
    img_data_array = np.random.permutation(img_data_array)

    


                

    return ( np.array(img_data_array[:up]), np.array(img_data_array[:up])), ( np.array(img_data_array[up:]), np.array(img_data_array[up:]))


(train_images, train_labels), (test_images, test_labels) = create_dataset(IMG_FOLDER,0.8)




# Przekształcanie wielkości obrazów do 64x64x1 pixel
train_images = train_images.reshape((len(train_images), IMG_HEIGHT, IMG_HEIGHT, 3))


test_images = test_images.reshape((len(test_images), IMG_HEIGHT, IMG_HEIGHT, 3))


y_train =train_images[:train_images.shape[0] - train_images.shape[0] % BATCH_SIZE]
x_train = np.expand_dims(np.arange(y_train.shape[0]), axis=1)
# Tworzenie modelu sieci
FIRST = 6
L=10
name = 11
if NEW == True:
    model = models.Sequential()

    # Dodanie pierwszej warstwy konwolucyjnej złożonej z 32 kerneli o wielkości 3x3
    
    model.add(layers.Conv2D(FIRST, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3)))
    model.add(layers.GaussianNoise(0.008))
    model.add(layers.Conv2D(FIRST, (17, 17), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    # Dodanie warstwy spłaszczającej dane 2D do 1D
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(L*32, activation='relu', name = "enc"))
    model.add(layers.BatchNormalization( name = "enc_norm"))
    model.add(layers.GaussianNoise(0.008))
    model.add(layers.Dense(FIRST*(250-16-2)**2, activation='relu', name = "dec"))
    model.add(layers.Reshape((IMG_HEIGHT-16-2, IMG_HEIGHT-16-2,FIRST), name = "dec0"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2DTranspose(6, (17, 17), activation='relu', name = "dec1"))
    model.add(layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', name = "dec2"))
    
    #model.add(layers.Activation("sigmoid",name ="last"))
    #Kompilacja modelu
    model.compile(optimizer=Adam(lr=0.001), loss=keras.losses.BinaryCrossentropy())
else:
    with open("model8.json", 'r') as plik:
        loaded_model_json = plik.read()
            
    model = model_from_json(loaded_model_json)
    model.load_weights("model8.h5")
    model.compile(optimizer=Adam(lr=0.001), loss=keras.losses.MeanSquaredError())
    face = model.predict(train_images[0:1])[0]*255
    face = np.array(face,dtype="uint8")
    img = np.reshape(face,(250,250,3))
    cv2.imshow("face",img)

plot_model(model, to_file='model'+str(name)+'.png', show_shapes=True,show_dtype=True)
# Uczenie modelu danymi
# epoch - liczba iteracji
# batch_size - liczba elemenów z danych treningowych branych podczas pojedyńczego przejścia funkcji uczącej
#history = model.fit(x_train, y_train,  epochs=69, batch_size=BATCH_SIZE, verbose=1)
from keras.models import Model
enc = Model(inputs=model.input,outputs=model.get_layer('enc_norm').output)
loss = []
loss_val = []
TRAIN_EPOCHS = 2000
for i in range(TRAIN_EPOCHS):
    history = model.fit(train_images,train_images,validation_data=(test_images,test_images),  epochs=1, batch_size=BATCH_SIZE, verbose=1)

    
    x_enc = enc.predict(train_images, batch_size=BATCH_SIZE)

    x_mean = np.mean(x_enc, axis=0)
    y_stdx_stds = np.std(x_enc, axis=0)
    x_cov = np.cov((x_enc - x_mean).T)
    e, v = np.linalg.eig(x_cov)
    random = [x_mean[j]+np.random.uniform(-2*y_stdx_stds[j],2*y_stdx_stds[j]) for j in range(len(x_mean))]
   
    loss.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    if(i%4 == 3):
        model_json = model.to_json()
        with open("model"+str(name)+".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model"+str(name)+".h5")
            print("Saved model to disk")
        dec = models.Sequential()
            
        dec.add(model.get_layer('dec'))
        dec.add(model.get_layer('dec0'))
        dec.add(model.get_layer('dec2'))
        dec.add(model.get_layer('dec3'))
       
        dec.add(model.get_layer('dec4'))
        #dec.add(model.get_layer('last'))
        dec.compile(optimizer=Adam(lr=0.01), loss='mse')
        face = dec.predict(np.reshape(random,(-1,L*32)) )[0]*255
        face = np.array(face,dtype="uint8")

        img = np.reshape(face,(250,250,3))
        cv2.imshow("face",img)


        face = model.predict(train_images[0:3])[0]*255
        face = np.array(face,dtype="uint8")
        img = np.reshape(face,(250,250,3))
        face = np.array(face,dtype="uint8")
       
        img = np.reshape(face,(250,250,3))
        cv2.imshow("face2",img)
        cv2.imwrite(str(name) +"face"+str(i)+".png",img)
        cv2.waitKey(1)
        print(loss)
        # wyświetlenie wykresu przedstawiającego historię uczenia sieci
        plt.plot(loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig("Plot" + str(name))

        plt.plot(loss_val)
        plt.title('model val_loss')
        plt.ylabel('val_losss')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper left')
        plt.savefig("Plot_val")

 
model_json = model.to_json()
with open("model9.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model9.h5")
print("Saved model to disk")
cv2.waitKey(0)