import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
x_data = pd.read_csv("C:/Users/KIIT/Desktop/Python/linearX.csv")
y_data = pd.read_csv("C:/Users/KIIT/Desktop/Python/linearY.csv")

# Preprocess data
x = x_data.values
y = y_data.values.reshape(-1, 1)

# Normalize the feature
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Add bias term (intercept)
x = np.hstack((np.ones((x.shape[0], 1)), x))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost function
def compute_cost(x, y, theta):
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient descent
def gradient_descent(x, y, theta, lr, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        h = sigmoid(np.dot(x, theta))
        gradient = (1 / m) * np.dot(x.T, (h - y))
        theta -= lr * gradient
        costs.append(compute_cost(x, y, theta))
    return theta, costs

# Initialize parameters
theta = np.zeros((x.shape[1], 1))
learning_rate = 0.1
iterations = 1000

# Perform gradient descent
theta, costs = gradient_descent(x, y, theta, learning_rate, iterations)

# Final cost and parameters
final_cost = costs[-1]
print(f"Final cost: {final_cost:.4f}, Parameters: {theta.ravel()}")

# Plot cost vs. iterations
plt.plot(range(len(costs)), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# Plot dataset with decision boundary
plt.figure(figsize=(8, 6))

# Scatter plot for data points
plt.scatter(x[:, 1], y, label="Data Points", color="blue")

# Decision boundary (sigmoid curve)
x_boundary = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
y_boundary = sigmoid(theta[0] + theta[1] * x_boundary)  # Sigmoid function for decision boundary
plt.plot(x_boundary, y_boundary, 'r', label='Decision Boundary')

plt.xlabel('Feature 1 (Normalized)')
plt.ylabel('Probability')
plt.title('Dataset with Decision Boundary')
plt.legend()
plt.grid()
plt.show()

# Test different learning rates
learning_rates = [0.1, 5]
all_costs = []
for lr in learning_rates:
    _, lr_costs = gradient_descent(x, y, np.zeros((x.shape[1], 1)), lr, 100)
    all_costs.append(lr_costs)

# Plot cost vs iterations for different learning rates
for lr, costs in zip(learning_rates, all_costs):
    plt.plot(range(len(costs)), costs, label=f'lr={lr}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Confusion matrix and metrics
def predict(x, theta):
    return (sigmoid(np.dot(x, theta)) >= 0.5).astype(int)

y_pred = predict(x, theta)

confusion_matrix = np.array([
    [(y_pred[y == 0] == 0).sum(), (y_pred[y == 0] == 1).sum()],
    [(y_pred[y == 1] == 0).sum(), (y_pred[y == 1] == 1).sum()]
])

accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Confusion Matrix:")
print(confusion_matrix)
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")
