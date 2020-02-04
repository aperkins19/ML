raw_name = input('Enter the name of the model to load: ')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import os, os.path
import cv2


from PIL import Image
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt


## predict function

def predict_image():
    img_name = input('Enter image to predict: ')
    img = img_name + '.png'
    def load(filename):
       np_image = Image.open(filename)
       np_image = np.array(np_image).astype('float32')/255
       np_image = transform.resize(np_image, (150, 150, 3))
       np_image = np.expand_dims(np_image, axis=0)
       return np_image

    image = load(img)

    pred = loaded_model.predict(image)
    pred_round = int(np.round(pred[0]))

    if pred_round == 0:
        print('I think this cell is infected')

    if pred_round == 1:
        print('I think this cell is not infected')

    im = Image.open(img)
    im.show()


# set up test dataset

image_width, image_height = 150, 150

validation_data_dir = 'data/val/'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 15
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_height, image_height)
else:
    input_shape = (image_height, image_height, 3)


# create generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')



# load model

os.chdir("C:/Users/User/Documents/Python_Projects/ML/Malaria/models")
name = raw_name+'.json'
weights_name = raw_name+"_weights.h5"
print(weights_name)
# load json and create model
json_file = open(name, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_name)
print("Loaded model from disk")
loaded_model.summary()


# compile and report accuracy
os.chdir("C:/Users/User/Documents/Python_Projects/ML/Malaria")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print()
print('Compiling...')
print()
print()
print()
print()

keep_going = 0
while keep_going == 0:
    predict_image()
    another = input('Enter y/n to predict another ')
    if another == 'n':
        keep_going =+ 1
    print()
