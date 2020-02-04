from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import os, os.path

# set up test dataset

img_width, img_height = 150, 150

train_data_dir = 'data_raw/'
validation_data_dir = 'data_raw/'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 15
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

test_datagen = ImageDataGenerator(rescale=1./255)

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


# Load model

raw_name = input('Enter the name of the model to load: ')
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
print()
print('Running test dataset...')
score = loaded_model.evaluate(test_generator, verbose=0)
print()
print()
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
