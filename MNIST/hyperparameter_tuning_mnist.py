import pandas as pd 
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


# you should get 60,000 training examples of 28x28 matrices and 10,000 test examples 
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#normalizing and flattening vectors
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# normalizing the data to help with the training
X_train_flat_norm = X_train_flat / 255
X_test_flat_norm = X_test_flat / 255

#one hot encoding y values
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train_hot = np_utils.to_categorical(y_train, n_classes)
y_test_hot = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train_hot.shape)

#building and compiling model using Keras
#in order to use keras with scikitlearn, have to wrap keras model in KerasClassifier class
#arguments for create_model function can be passed to 

learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
optimizer = ['adam', 'SGD', 'RMSprop']
neurons = [10, 20, 50, 100, 200]

param_grid = dict(learn_rate=learn_rate, momentum=momentum, optimizer=optimizer, neurons=neurons)

#creating function to pass to RandomizedSearchCV
def create_model(optimizer='adam', learn_rate=0.01, momentum=0, neurons=128):
    #create model
    model=Sequential()
    model.add(Dense(neurons, input_shape=(784,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    if optimizer=='adam':
        optimizer=adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=10**-8)
    if optimizer=='SGD':
        optimizer=SGD(lr=learn_rate, momentum=momentum)
    if optimizer=='RMSprop':
        optimizer=RMSprop(lr=learn_rate, rho=0.9, epsilon=10**-8)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model 

#batch size and epochs can also be tuned
#need to use KerasClassifier() to turn Keras object into something sklearn can use 
model=KerasClassifier(build_fn=create_model, batch_size=32, verbose=1) 

#creating grid object, samples parameters 10 times, default cv=kfold(3), n_jobs=1 means it's only running on one core
grid=RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=20, n_iter=10, n_jobs=1)
grid_result=grid.fit(X_train_flat_norm, y_train_hot)

#trying with all cores
model=KerasClassifier(build_fn=create_model, batch_size=32, verbose=1) 
grid=RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=20, n_iter=10, n_jobs=-1)
grid_result=grid.fit(X_train_flat_norm, y_train_hot)

#pulling results
grid_result_df = pd.DataFrame(grid_result.cv_results_)
grid_result.best_score_
grid.best_params_
grid.best_estimator_






