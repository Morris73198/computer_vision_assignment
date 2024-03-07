import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function and its gradient
def objective_function(A, b, x, y):
    F1 = A[0][0]*x + A[0][1]*y - b[0]
    F2 = A[1][0]*x + A[1][1]*y - b[1]
    return F1**2 + F2**2

def gradient(A, b, x, y):
    F1 = A[0][0]*x + A[0][1]*y - b[0]
    F2 = A[1][0]*x + A[1][1]*y - b[1]
    df_dx = 2 * (A[0][0] * F1 + A[1][0] * F2)
    df_dy = 2 * (A[0][1] * F1 + A[1][1] * F2)
    return df_dx, df_dy

# Gradient Descent Algorithm
def gradient_descent(A, b, learning_rate, num_iterations, initial_x, initial_y):
    x_history = []
    y_history = []
    z_history = []

    x, y = initial_x, initial_y

    for _ in range(num_iterations):
        df_dx, df_dy = gradient(A, b, x, y)
        x = x - learning_rate * df_dx
        y = y - learning_rate * df_dy
        z = objective_function(A, b, x, y)

        x_history.append(x)
        y_history.append(y)
        z_history.append(z)

    return x_history, y_history, z_history

# Given values
b = np.array([[1], [2]])
A = np.array([[1, -2], [1, -2]])
learning_rate = 0.1
num_iterations = 50
initial_x, initial_y = -2, 2

# Run gradient descent
x_history, y_history, z_history = gradient_descent(A, b, learning_rate, num_iterations, initial_x, initial_y)

# Meshgrid for plotting the objective function
xx, yy = np.meshgrid(np.arange(-2, 2.1, 0.1), np.arange(-2, 2.1, 0.1))
zz = objective_function(A, b, xx, yy)

# Plot the entire 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the entire 3D surface
ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.5, label='Objective Function Surface')

# Plot the gradient descent path
ax.scatter(x_history, y_history, z_history, color='red', label='Gradient Descent Path')
ax.plot(x_history, y_history, z_history, color='blue', linewidth=2, label='Optimal Solutions Line')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Objective Function')
ax.set_title('Gradient Descent and Optimal Solutions')

# Show the plot
plt.legend()
plt.show()
