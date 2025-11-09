import numpy as np

class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs

    def fit(self, X, y):
        # Convert labels to -1 or 1
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # Initialize weights
        self.b = 0  # Initialize bias term

        # Training loop (epochs)
        for _ in range(self.epochs):
            # Loop through each sample to perform the update
            for i in range(n_samples):
                x_i = X[i]
                y_i = y_[i]
                
                # Compute the decision function (w * x_i - b)
                z = 0
                for j in range(n_features):
                    z += self.w[j] * x_i[j]
                z -= self.b  # Subtract the bias term

                # Check if the condition is violated (margin)
                if y_i * z >= 1:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param * self.w - y_i * x_i
                    db = y_i

                # Update weights and bias manually
                for j in range(n_features):
                    self.w[j] -= self.lr * dw[j]
                self.b -= self.lr * db

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            z = 0
            for j in range(X.shape[1]):
                z += self.w[j] * X[i][j]
            z -= self.b  # Subtract bias term
            predictions.append(np.sign(z))  # Return 1 or -1 based on the sign

        return np.array(predictions)

# Example usage:
if __name__ == "__main__":
    # Example data (2D)
    X = np.array([[2, 3], [1, 5], [2, 8], [6, 8], [5, 2], [7, 3]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Train the SVM
    model = LinearSVM(lr=0.01, lambda_param=0.01, epochs=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
