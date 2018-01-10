#Trains a simple deep NN on the MNIST dataset.

from __future__ import print_function

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

start = datetime.datetime.now()

batch_size = 128
num_classes = 10

####################################################################
# PARAMETERS WE WILL ADJUST IN THIS EXAMPLE                        #
####################################################################
epochs = 3  # an epoch is a single run of all data in training set
max_train_length =  500 # used to limit the number of training samples
neurons = 32  # number of neurons in each hidden layer

# Updating the parameters to these values will give you an accuracy around 98%
#epochs = 10  # an epoch is a single run of all data in training set
#max_train_length =  60000 # used to limit the number of training samples
#neurons = 512  # number of neurons in each hidden layer


####################################################################
####################################################################
# STEP 1: LOAD IMAGE DATA                                          #
####################################################################
####################################################################

# the data, shuffled and split between train and test sets
# This function loads 60,000 training samples (x_train, y_train) and 10,000 test samples (x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# limit the training set to a specified size
x_train = x_train[0:max_train_length]
y_train = y_train[0:max_train_length]

####################################################################
####################################################################
# STEP 2: PRE-PROCESS IMAGE DATA                                   #
####################################################################
####################################################################

# Reshape single image data from 2d array to 1d
# simplified example using 7x7 istead of 28x28 matrix
#[[0 0 0 0 0 0 0]   =>  [0 0 0 0 0 0 0 0 4 0 0 4 0 0 0 4 0 0 4 0 0 0 4 4 4 4 0 0 0 0 0 0 4 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0]
# [0 4 0 0 4 0 0]
# [0 4 0 0 4 0 0]
# [0 4 4 4 4 0 0]
# [0 0 0 0 4 0 0]
# [0 0 0 0 4 0 0]
# [0 0 0 0 0 0 0]]

x_train = x_train.reshape(max_train_length, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#rescale pixel values from 0-255 to 0-1
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

####################################################################
####################################################################
# STEP 3: SET UP THE MODEL AND SPECIFY LAYERS AND OPTIONS          #
####################################################################
####################################################################

#Sequential model represents a linear stack of layers
model = Sequential()

# Densely-connected neural network layer
# First layer specifies the input layer and the first hidden layer
# The 3 hidden layers use the rectified linear unit (ReLU) for activation - ReLU is commonly used in deep learning hidden layers
model.add(Dense(neurons, activation='relu', input_shape=(784,)))

#Dropout can help prevent overfitting - this will drop 20% of input units - setting weight to 0
model.add(Dropout(0.2))

# Second hidden layer uses the output of the first layer as input
model.add(Dense(neurons, activation='relu'))
model.add(Dropout(0.2))

# Third hidden layer
model.add(Dense(neurons, activation='relu'))
model.add(Dropout(0.2))

# output layer will have 10 neurons since we are classifying 10 digits
# the output layer uses the softmax activation function instead of ReLU.  This is better suited for classification
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

####################################################################
####################################################################
# STEP 4: FIT THE MODEL WITH TRAINING DATA                         #
####################################################################
####################################################################

# fit the model with train data (x) and labels (y)
# validation_data is used to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

####################################################################
####################################################################
# STEP 5: DETERMINE ACCURACY OF TRAINED MODEL AGAINST TEST DATA    #
####################################################################
####################################################################

score = model.evaluate(x_test, y_test, verbose=0)
end = datetime.datetime.now()
elapsed = end - start

print('\nExecution Time =', elapsed.total_seconds(), 'seconds')
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100, '%')

