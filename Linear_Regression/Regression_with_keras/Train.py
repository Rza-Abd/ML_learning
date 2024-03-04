import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)

# Creat random Data
x = np.linspace(0, 50, 51)
y = x + np.random.rand(51) * 10

# Define Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=[1]))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(x, y, epochs=100)

plt.plot(history.history['loss'])
plt.show()

model.save_weights('Regression_model.h5')