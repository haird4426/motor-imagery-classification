#%%
import numpy as np
import importlib 
import matplotlib.pyplot as plt
import preprocessing
importlib.reload(preprocessing)
from preprocessing import *
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Conv1D,Conv2D,MaxPooling1D,Flatten,Dense,Dropout,BatchNormalization, GRU, LSTM, RNN
from keras import regularizers as reg
from keras.callbacks import TensorBoard

#%%
def cnn_plot(conv_layers=3,conv_sizes=(64,128,256),filter_size=3, fc_layers=2,fc_sizes=(4096,2048),
        dropout=0.5,pool_size=2,init='he_uniform',act='relu',optim='adam',pool=True,
        reg = reg.l2(0.05),epochs=10):

    model = Sequential()
    for i in range(conv_layers):
        #畳み込み層
        model.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              activation = act,kernel_initializer=init,kernel_regularizer=reg))
        #model.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              #activation = act,kernel_initializer=init,kernel_regularizer=reg))
        #model.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              #activation = act,kernel_initializer=init,kernel_regularizer=reg))
        
        model.add(BatchNormalization())
        if pool:
            model.add(MaxPooling1D(pool_size = 2))  #プーリング層
    model.add(Flatten())
    for j in range(fc_layers):
        model.add(Dense(fc_sizes[j], activation = act,kernel_initializer=init,kernel_regularizer=reg))
        model.add(Dropout(dropout))
    model.add(Dense(4, activation = 'softmax',kernel_initializer=init))
    model.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs,batch_size=64)
    
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

#%%
def nn_test(conv_layers=3,conv_sizes=(64,128,256),filter_size=3, fc_layers=2,fc_sizes=(4096,2048),
        dropout=0.5,pool_size=2,init='he_uniform',act='relu',optim='adam',pool=True,
        reg = reg.l2(0.05),epochs=10):
    model = Sequential()
    for i in range(conv_layers):
        model.add(Dense(conv_sizes[i],activation="relu",input_shape = X_train.shape[1:]))

    model.add(Dropout(0.5))

    model.add(Dense(4,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30)

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


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
#cnn_plot()
nn_test()

#%%
#X, y = import_data_test(every=False)
X, y = import_data(every=False)
#print(X)
X_train,X_test,y_train,y_test = train_test_total(X, y) 
#↑をモデル作成内でやればモジュール化できるかも

axis_x = []
axis_y = X_train[3][0]
for i in range(22):
    axis_x.append(i)
print(axis_x)
print(axis_y)
plt.scatter(axis_x,axis_y)
plt.show()

#print(X_train.dtype)   #float64
#print(X_train.shape) #(2508, 1000, 22)
#print(X_test.shape)  #(50, 1000, 22)
#print(y_train.shape) #2508
#print(y_test.shape)   #50

#print(X)        
#print(X_train) 　#3次元リスト 
#print(X_test)  #3次元リスト

#print(y)       
#print(y_train)  #0,1,2,3のいずれか　１次元リスト
#print(y_test)    #0,1,2,3のいずれか　１次元リスト 

cnn_plot(conv_layers=3,conv_sizes=(32,32,32),fc_layers=2,fc_sizes=(512,256),epochs=10)


#%%
"""
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn_plot(conv_layers=2,conv_sizes=(32,32,32),fc_layers=3,fc_sizes=(1024,512,256),epochs=50)
"""

#%%
"""
def cnn(conv_layers=3,conv_sizes=(64,128,256),filter_size=3, fc_layers=2,fc_sizes=(4096,2048),
        dropout=0.5,pool_size=2,init='he_uniform',act='relu',optim='adam',pool=True,
        reg = reg.l2(0.05)):

    model = Sequential()
    for i in range(conv_layers):
        model.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              activation = act,kernel_initializer=init,kernel_regularizer=reg))
        model.add(BatchNormalization())
        if pool:
            model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    for j in range(fc_layers):
        model.add(Dense(fc_sizes[j], activation = act,kernel_initializer=init,kernel_regularizer=reg))
        model.add(Dropout(dropout))
    model.add(Dense(4, activation = 'softmax',kernel_initializer=init))
    model.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10,batch_size=64)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=2,conv_sizes=(64,128),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=True)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=2,conv_sizes=(64,128),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=3,conv_sizes=(64,128,256),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y, train_all=False)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),dropout=0.1)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),dropout=0.9)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),pool=False)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),act='tanh')


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y,standardize=False)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y,standardize=False)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=3,conv_sizes=(64,64,64),fc_layers=2,fc_sizes=(512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=3,conv_sizes=(32,32,32),fc_layers=2,fc_sizes=(512,256))


#%%
X, y = import_data(every=True)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=True)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=3,conv_sizes=(64,128,256),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=True)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
#Redefine CNN with the batchnorm commented 
# This is without batch norm
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
#Redefine CNN with the batchnorm commented 
# This is without batch norm
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),pool=False)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),act='sigmoid')
"""