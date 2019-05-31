import matplotlib.pyplot as plt
import numpy as np
import importlib 
from preprocessing import *
import preprocessing
importlib.reload(preprocessing)
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Conv1D,Conv2D,MaxPooling1D,Flatten,Dense,Dropout,BatchNormalization,GRU
from keras import regularizers as regu
#fc_layers=3
#fc_sizes=(1024,512,256)
#dropout=0.5
#pool_size=2
#act='relu'
#optim='adam'
#pool=True

def data_set():
    X, y = import_data(every=False)
    X_train,X_test,y_train,y_test = train_test_total(X, y)
    
    return X_train,X_test,y_train,y_test

def test(X_train,X_test,y_train,y_test):
    conv_layers=2
    conv_sizes=(64,128)
    filter_size=3
    fc_layers=3
    fc_sizes=(1024,512,256)
    dropout=0.5
    #pool_size=2
    init='he_uniform'
    act='relu'
    optim='adam'
    pool=True
    reg = regu.l2(0.05)
    epochs=30
    
    classifier = Sequential()
    for i in range(conv_layers):
        classifier.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(BatchNormalization())
        if pool:
            classifier.add(MaxPooling1D(pool_size = 2))
    classifier.add(Flatten())
    for j in range(fc_layers):
        classifier.add(Dense(fc_sizes[j], activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(Dropout(dropout))
    classifier.add(Dense(4, activation = 'softmax',kernel_initializer=init))
    classifier.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = classifier.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs,batch_size=64)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


"""
def cnn(input_shape_0,input_shape_1):    
    input_shape = (1000,22)
    filter_size=3
    conv_layers=2
    conv_sizes=(64,64)
    reg = regu.l2(0.05)
    
    model = Sequential()
    for i in range(conv_layers):
        model.add(Conv1D(conv_sizes[i], filter_size, input_shape = input_shape,
                              activation='relu',kernel_initializer='he_uniform',kernel_regularizer=reg))
        model.add(BatchNormalization())
        #if pool:
        #    model.add(MaxPooling1D(pool_size))
    model.add(Flatten())
    #for j in range(fc_layers):
        #model.add(Dense(fc_sizes[j], activation = act,kernel_initializer='he_uniform',kernel_regularizer=reg))
        #model.add(Dropout(dropout))
    model.add(Dense(4, activation = 'softmax',kernel_initializer='he_uniform'))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    # classifier.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10,batch_size=64)

    return model
"""
"""
def create_cnn(input_shape,dropout,print_summary):

    # basis of the CNN_STFT is a Sequential network
    model = Sequential()

    model.add(Conv1D(filters = 24, kernel_size = (12, 12),
                         strides = (1, 1), name = 'conv1',
                         border_mode = 'same'))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding = 'valid',
                               data_format = 'channels_last'))

    classifier.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                        activation = act,kernel_initializer=init,kernel_regularizer=reg))
"""