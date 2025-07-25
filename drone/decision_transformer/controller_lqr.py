import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, A, B, Q=None, R=None):
        n = A.shape[0]
        m = B.shape[1]
        self.Q = np.eye(n) if Q is None else Q
        self.R = np.eye(m) if R is None else R
        P = solve_continuous_are(A, B, self.Q, self.R) # Solve Riccati equation used by LQR
        self.K = np.linalg.inv(self.R) @ B.T @ P # Computes LQR gain

    def get_control(self, x, x_target):
        return -self.K @ (x - x_target) # This is the linear feedback
