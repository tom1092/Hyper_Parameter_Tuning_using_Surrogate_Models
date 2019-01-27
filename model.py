# Reproducibility
import numpy as np
from tensorflow import set_random_seed
import random as rn

np.random.seed(17)
set_random_seed(17)
rn.seed(17)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization


class Model:

    def __init__(self, x_train, y_train, x_val, y_val, batch_size, epochs):

        self.batch_size = batch_size
        self.epochs = epochs
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val

    # This method define the NN model
    def create_model(self, regularizer, lr, momentum, decay, drop):
        model = Sequential()

        model.add(Dense(20, input_dim=60, kernel_initializer=keras.initializers.glorot_normal()))
        model.add(Activation('relu'))
        model.add(Dropout(drop))

        model.add(Dense(20, kernel_initializer=keras.initializers.glorot_normal()))
        model.add(Activation('relu'))

        model.add(Dense(20, kernel_initializer=keras.initializers.glorot_normal()))
        model.add(Activation('relu'))

        model.add(Dense(1, kernel_regularizer=keras.regularizers.l1(regularizer)))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

        opt = keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True, decay=decay)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
        return model

    def evaluate_model(self, x):

        regularizer, lr, momentum, decay, dropout = [x[i] for i in range(len(x))]
        model = self.create_model(regularizer, lr, momentum, decay, dropout)

        print ("\n### EVALUATING MODEL WITH PARAMS ###")
        print ("Regularizer l1: "+str(regularizer))
        print ("Learning rate: " + str(lr))
        print ("Momentum: " + str(momentum))
        print ("Decay: " + str(decay))
        print ("Dropout: " + str(dropout))
        history = model.fit(x=self.x_train, y=self.y_train, validation_data=(self.x_val, self.y_val), batch_size=self.batch_size, epochs=self.epochs, verbose=0,
                            shuffle=False)

        # We want to minimize the val loss at the end of the training

        val = history.history['val_loss'][self.epochs - 1]

        print ("Validation loss: "+str(val))
        return history, val
