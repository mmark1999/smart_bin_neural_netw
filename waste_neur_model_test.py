import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

x_test = []

pickle_in = open("X.pickle","rb")
x_train = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y_train = pickle.load(pickle_in)

"""
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
"""

x_train = tf.keras.utils.normalize(x_train, axis=0)

X_train = np.array(x_train)
y_train = np.array(y_train)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

"""
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

"""

model.save('waste.model')

new_model = tf.keras.models.load_model('waste.model')

predictions = new_model.predict(x_train)

#print(predictions)

#print(np.argmax(predictions[0]))
print(predictions[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
