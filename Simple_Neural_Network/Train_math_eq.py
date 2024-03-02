import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Math Eq : y = 3X^2 + 2X + 1
# Generate some random data
np.random.seed(42)
x = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = 3 * x ** 2 + 2 * x + 1

# Build Neural Network
model = Sequential([
    Dense(10, input_shape=(1,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(x, y, epochs=100, batch_size=32, verbose=1)

# Save model
model.save_weights('math_eq_model.h5')