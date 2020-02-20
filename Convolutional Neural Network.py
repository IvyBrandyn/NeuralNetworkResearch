# Ivy Vasquez Sandoval
"""
This is representative of independent research into Neural Networks
exploring how these models are created, initialized, used, etc.
"""

# Verify GPU memory growth
import tensorflow as tf
config = tf.config.experimental.list_physical_devices('GPU')
for device in config:
    tf.config.experimental.set_memory_growth(device, True)

# Create the Covolutional Neural Network
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Flatten, Dense, Dropout

# Initilize CNN Model
CNN_Model = Sequential()

# Convolution and pooling layers
CNN_Model.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
CNN_Model.add(MaxPooling2D(pool_size = (2, 2)))
CNN_Model.add(Conv2D(32, (3, 3), activation = 'relu'))
CNN_Model.add(MaxPooling2D(pool_size = (2, 2)))
CNN_Model.add(Conv2D(32, (3, 3), activation = 'relu'))
CNN_Model.add(MaxPooling2D(pool_size = (2, 2)))
CNN_Model.add(Conv2D(32, (3, 3), activation = 'relu'))
CNN_Model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening data
CNN_Model.add(Flatten())

# Fully connected layers with Dropout
CNN_Model.add(Dense(units = 64, activation = 'relu'))
CNN_Model.add(Dropout(rate = 0.4))
CNN_Model.add(Dense(units = 64, activation = 'relu'))
CNN_Model.add(Dropout(rate = 0.2))
CNN_Model.add(Dense(units = 64, activation = 'relu'))
CNN_Model.add(Dropout(rate = 0.1))
CNN_Model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling Convolutional Neural Network
CNN_Model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image Augmentation to avoid overfitting
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

CNN_Model.fit_generator(training_set,
                         steps_per_epoch = 8000/batch_size,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000/batch_size)

# Making new prediction
import numpy as np
from tensorflow.python.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/Cat or dog2.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = CNN_Model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)