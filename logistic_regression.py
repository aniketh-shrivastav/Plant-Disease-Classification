import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        # Manual implementation of the sigmoid function
        return 1 / (1 + np.exp(-z))  # Equivalent to 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        # Add bias term (column of ones) to the feature matrix X
        X_bias = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.weights = np.zeros(X_bias.shape[1])

        for _ in range(self.epochs):
            # Calculate linear combination (z = X * w)
            z = np.zeros(X_bias.shape[0])
            for i in range(X_bias.shape[0]):  # Iterating through each sample
                for j in range(X_bias.shape[1]):  # Iterating through each feature + bias
                    z[i] += X_bias[i, j] * self.weights[j]
            
            # Compute the sigmoid of z
            predictions = np.zeros(z.shape)
            for i in range(len(z)):
                predictions[i] = 1 / (1 + np.exp(-z[i]))  # Sigmoid function
            
            # Compute gradient (error * input)
            gradient = np.zeros(X_bias.shape[1])
            for j in range(X_bias.shape[1]):  # Loop over each feature
                grad = 0
                for i in range(X_bias.shape[0]):
                    grad += (predictions[i] - y[i]) * X_bias[i, j]
                gradient[j] = grad / X_bias.shape[0]
            
            # Update the weights using gradient descent
            self.weights -= self.lr * gradient

    def predict(self, X):
        # Add bias term (column of ones) to the feature matrix X
        X_bias = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        predictions = np.zeros(X_bias.shape[0])

        # Linear combination (z = X * w)
        for i in range(X_bias.shape[0]):
            z = 0
            for j in range(X_bias.shape[1]):
                z += X_bias[i, j] * self.weights[j]

            # Apply sigmoid function
            predictions[i] = 1 / (1 + np.exp(-z))  # Sigmoid function

        # Convert to binary prediction (0 or 1)
        return (predictions >= 0.5).astype(int)

# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    X = np.array([[2, 3], [1, 5], [2, 8], [6, 8], [5, 2], [7, 3]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Create the Logistic Regression model
    model = LogisticRegressionScratch(lr=0.1, epochs=1000)

    # Train the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    print("Predictions:", predictions)
