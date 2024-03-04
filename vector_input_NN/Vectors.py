import matplotlib.pyplot as plt
import numpy as np

# Create a single vector
input_vector = np.array([1.72, 1.23])
weight_1 = np.array([1.26, 0])
weight_2 = np.array([2.17, 0.32])

fig, ax = plt.subplots()
ax.quiver(0,0, input_vector[0], input_vector[1], angles='xy', scale_units='xy', scale=1, color='r')
ax.quiver(0,0, weight_1[0], weight_1[1], angles='xy', scale_units='xy', scale=1, color='b')
ax.quiver(0,0, weight_2[0], weight_2[1], angles='xy', scale_units='xy', scale=1, color='g')

ax.set_xlim([0, 2.5])
ax.set_ylim([0, 2.5])

plt.grid()
plt.show()

# Dot product
mult_input_w1_manual = input_vector[0] * weight_1[0] + input_vector[1] * weight_1[1]
print(mult_input_w1_manual)

mult_input_w1_np = np.dot(input_vector, weight_1)
print(mult_input_w1_np)

mult_input_w2_np = np.dot(input_vector, weight_2)
print(mult_input_w2_np)