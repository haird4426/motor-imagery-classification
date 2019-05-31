#%%
#from CNN_test import *
from CNN_test import data_set,cnn_test

import numpy as np
import importlib 
import preprocessing
importlib.reload(preprocessing)
from preprocessing import *
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Conv1D,Conv2D,MaxPooling1D,Flatten,Dense,Dropout,BatchNormalization, GRU, LSTM, RNN
from keras import regularizers as reg

#%%
#set parameter
DROPOUT = 0.2   # dropout rate in float

#%%
#X, y = import_data(every=False)
#print(X)
#X_train,X_test,y_train,y_test = train_test_subject(X, y)
#print(X_train)
#print(X_train.shape[1:]) #(1000,22)
#print(X_train.shape[1:][1]) #22

#input_shape = list(X_train.shape[1:]) 
#print(input_shape) #[1000,22]

X_train,X_test,y_train,y_test = data_set()
cnn_test(X_train,X_test,y_train,y_test)

#%%
"""
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
cnn_plot(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn_plot(conv_layers=3,conv_sizes=(32,32,32),fc_layers=2,fc_sizes=(512,256),epochs=30)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn_plot(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),epochs=30)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),pool=False)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256),act='sigmoid')
"""