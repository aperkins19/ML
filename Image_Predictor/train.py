from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import os, os.path
#####

# dimensions of the input images.
image_width = 150
image_height = 150
train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 15
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_width, image_height)
else:
    input_shape = (image_width, image_height, 3)

########

#get model

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

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
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the source directory
        target_size=(image_width, image_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # binary labels for binary_crossentropy loss function,

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')


early_stopping_monitor = EarlyStopping(patience=2)

model.fit_generator(train_generator,validation_data=validation_generator,  epochs=epochs, callbacks=[early_stopping_monitor])


###### save weights?

# Create a callback that saves the model's weights
def save_model():
    save_weights_question = input('Do you want to save these weights?  Type: "y or n" and hit enter.   ')
    print()
    raw_name = input('Name the model: ')
    name = raw_name+'.json'
    weights_name = raw_name+"_weights.h5"
    print(weights_name)

    if save_weights_question == 'y':
        os.chdir("C:/Users/User/Documents/Python_Projects/ML/Malaria/models")
        # counts number of weights already existing for filename
        working_model = model.to_json()
        with open(name, "w") as json_file:
            json_file.write(working_model)
        # serialize weights to HDF5
        model.save_weights(weights_name)
        print("Saved model: "+str(name)+" to disk")

save_model()
