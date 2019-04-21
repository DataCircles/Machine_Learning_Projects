

# you should get 60,000 training examples of 28x28 matrices and 10,000 test examples 
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()