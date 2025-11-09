import numpy as np

class KernelSVM:
    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma=0.05, max_iters=500):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.max_iters = max_iters

    # ---------------------------
    # Kernel functions
    # ---------------------------
    def _kernel(self, x1, x2):
        if self.kernel == "linear":
            return np.dot(x1, x2)
        elif self.kernel == "poly":
            return (1 + np.dot(x1, x2)) ** self.degree
        elif self.kernel == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel type")

    # ---------------------------
    # Training using simplified SMO
    # ---------------------------
    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)       # convert class labels to {-1, +1}
        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Precompute kernel matrix (expensive for large datasets)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])

        for _ in range(self.max_iters):
            for i in range(n_samples):
                j = np.random.randint(0, n_samples - 1)
                if j >= i:
                    j += 1

                # Predictions
                yi = np.sum(self.alpha * y * K[:, i]) + self.b
                yj = np.sum(self.alpha * y * K[:, j]) + self.b

                Ei = yi - y[i]
                Ej = yj - y[j]

                # Old alpha values
                alpha_i_old = self.alpha[i]
                alpha_j_old = self.alpha[j]

                # Compute bounds (L, H)
                if y[i] != y[j]:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                if L == H:
                    continue

                # Compute eta
                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Update alpha_j
                self.alpha[j] -= y[j] * (Ei - Ej) / eta
                # clip values
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                # Update alpha_i
                self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                # Bias update
                b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                     y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]

                b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                     y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                self.b = (b1 + b2) / 2

        self.X_sv = X
        self.y_sv = y
        print("Training completed — Support vectors:", np.count_nonzero(self.alpha > 0))

    # ---------------------------
    # Make a prediction
    # ---------------------------
    def predict(self, X):
        y_pred = []
        for x in X:
            s = 0
            for alpha, y_sv, x_sv in zip(self.alpha, self.y_sv, self.X_sv):
                if alpha > 0:
                    s += alpha * y_sv * self._kernel(x_sv, x)
            s += self.b
            y_pred.append(np.sign(s))
        return np.array(y_pred)
