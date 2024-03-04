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

# Load the Model
model.load_weights('Regression_model.h5')
model.summary()

y_pred = model.predict(x)

# plt.scatter(x, y, label='Training Data')
# plt.plot(x, y_pred, label="Predict With Model", color='c')
# plt.legend()
# plt.show()

layer = model.get_layer(index=0)
print(layer)

weights = model.get_weights()
print(weights)

w, b = weights[0][0].item() , weights[1].item()
print(f'W: {w}, b: {b}')
