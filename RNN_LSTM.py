
#%%
import tensorflow as tf
import numpy as np
import importlib 
import preprocessing
importlib.reload(preprocessing)
from preprocessing import *
import warnings
warnings.filterwarnings('ignore')


#%%
X, y = import_data()
X_train,X_test,y_train,y_test = train_test_total(X, y)


#%%
def rnn_model(features, labels, mode, params):
  
  num_hidden_layers = len(params['hidden_layers'])  
  
  input_layer = features["x"]    
  cell1 = tf.nn.rnn_cell.LSTMCell(name = 'cell1', num_units = params['hidden_layers'][0])
  outputs, state = tf.nn.dynamic_rnn(cell=cell1,
                                   inputs=input_layer,
                                   dtype=tf.float64)
  outputs = tf.nn.relu(outputs)      
  outputs = tf.layers.dropout(inputs=outputs, rate=0.5, training=(mode==tf.estimator.ModeKeys.TRAIN))
  
  for i in range(num_hidden_layers-1):  
      cell = tf.nn.rnn_cell.LSTMCell(name = 'cell'+str(i+2),num_units = params['hidden_layers'][i+1])
      outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=outputs,
                                       dtype=tf.float64)
      outputs = tf.nn.relu(outputs)
  
  #FLatten the output of LSTM layers
  outputs = tf.contrib.layers.flatten(outputs) 

  # FC Layer
  logits = tf.layers.dense(inputs=outputs, units=params['num_classes'])

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  tf.summary.scalar('loss', loss)
      

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
  


#%%
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [32, 32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=64,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


#%%
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [32, 32, 32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


#%%
X, y = import_data(every=True)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [32, 32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [32, 32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [64, 64], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [32, 32, 32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [64, 64, 64], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [128, 128], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [64,64], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, model_dir="/model/", 
                                        params = {'hidden_layers' : [32,32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_subject(X, y,train_all=False)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, 
                                        params = {'hidden_layers' : [64,64], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y,standardize=False)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, 
                                        params = {'hidden_layers' : [64,64], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, 
                                        params = {'hidden_layers' : [32,32,32,32], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)


#%%
X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
eeg_classifier = tf.estimator.Estimator(model_fn=rnn_model, 
                                        params = {'hidden_layers' : [64], 'num_classes' : 4, 'learning_rate' : 0.001})
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=20,
    shuffle=True)

eeg_classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])

eval_train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
eval_train_results = eeg_classifier.evaluate(input_fn=eval_train_fn)

eval_test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
eval_test_results = eeg_classifier.evaluate(input_fn=eval_test_fn)
print('Train results are:',eval_train_results)
print('Test results are:',eval_test_results)
