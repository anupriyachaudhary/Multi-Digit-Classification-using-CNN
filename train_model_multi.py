import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.optimizers import Adam, RMSprop

import h5py
import numpy as np
from keras.utils.np_utils import to_categorical

def get_one_hot_labels(labelArray):
    size = labelArray.shape[0]
    
    train_labels = np.zeros((size, 11*4+5+2), dtype=int)
    train_labels[:,:11] = to_categorical(labelArray[:,0], 11)
    train_labels[:,11:22] = to_categorical(labelArray[:,1], 11)
    train_labels[:,22:33] = to_categorical(labelArray[:,2], 11)
    train_labels[:,33:44] = to_categorical(labelArray[:,3], 11)
    train_labels[:,44:49] = to_categorical(labelArray[:,4], 5)
    
    return train_labels

def load_data():
    # Open the file as readonly
    h5py_train = h5py.File('data/train.h5', 'r')
    h5py_test = h5py.File('data/test.h5', 'r')

    # Load the training, test and validation set
    X_train = h5py_train['train_dataset'][:]
    y_train = h5py_train['train_labels'][:]
    X_test = h5py_test['test_dataset'][:]
    y_test = h5py_test['test_labels'][:]

    print('Training set', X_train.shape, y_train.shape)
    print('Test set', X_test.shape, y_test.shape)
    
    scalar = 1 / 255.
    X_train = X_train * scalar
    X_test = X_test * scalar
    
    h5py_train.close()
    h5py_test.close()
    
    #train labels
    one_hot_train = get_one_hot_labels(y_train)
    one_hot_test = get_one_hot_labels(y_test)
    
    return (X_train,one_hot_train,X_test,one_hot_test)

def model_own():
    x = Input(shape=(64, 64, 3))
    
    #conv layer
    x = Conv2D(64, (3,3), activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same") (x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same") (x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(192, (3,3), activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same") (x)
    x = Dropout(0.1)(x)
    
    
    x = Conv2D(256, (3,3), activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same") (x)
    x = Dropout(0.1)(x)
    
    
    x = Conv2D(256, (3,3), activation='relu', padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="same") (x)
    x = Dropout(0.1)(x)
    
    #linear layer
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)

    digit1 = Dense(11, activation='softmax')(x)
    digit2 = Dense(11, activation='softmax')(x)
    digit3 = Dense(11, activation='softmax')(x)
    digit4 = Dense(11, activation='softmax')(x)
    num_digits = Dense(5, activation='softmax')(x)
    
    output = [digit1, digit2, digit3, digit4, num_digits]

    model = Model(inputs = x, outputs = output)

    model.compile(keras.optimizers.RMSprop(lr=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model


def model_train(model,X_train, y_train, X_test, y_test):
    batch_size = 128
    num_epoch = 5

    model_log = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              verbose=1,
              validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model_log

def output(model_log,name):
    model.save_weights("%s.h5" % name)
    print("Saved model to disk")


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    model = model_own()
    
    y_train = [y_train[:,:11], y_train[:,11:22], y_train[:,22:33], y_train[:,33:44], y_train[:,44:49]]
    y_test = [y_test[:,:11], y_test[:,11:22], y_test[:,22:33], y_test[:,33:44], y_test[:,44:49]]
    
    model_log = model_train(model,X_train, y_train, X_test, y_test)
    
    output(model_log,'model_own')
