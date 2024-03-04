import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(101)

x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x)

fig = plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data')
plt.show()

# Define Model

X = tf.placeholder('float')
Y = tf.placeholder('float')

# y = w*x + b
w = tf.Variable(np.random.rand(), name='W')
b = tf.Variable(np.random.rand(), name='b')

learning_rate = 0.01
epochs = 1000

y_pred = tf.add(b, tf.multiply(w, x))

# Mean Squard Erroe Cost Function
cost = tf.reduce_sum(tf.pow(y_pred - y , 2)) / (2 * n)

# Gradient Decent Optimizer
optimizer = tf.train.GradientDecentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for (_x, _y) in (X, Y):
            sess.run(optimizer, feed_dict = {X:_x, Y:_y})

        if (epoch + 1) % 50 == 0:
            c = sess.run(cost, feed_dict = {X:_x, Y:_y})
            print("Epoch", (epoch + 1, ": cost = ", c, "W = ", sess.run(w), "b = ", sess.run(b)))  