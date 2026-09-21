import numpy as np
from scipy.special import expit

class CustomLink:
    @staticmethod
    def evaluate_z(a, b):
        """z_i(b) = log(expit(a + b))"""
        z = -np.log1p(np.exp(-(a + b)))

        # print("z info:")
        # print(np.min(z), np.max(z), np.mean(z))
        return z
    
    @staticmethod
    def evaluate_z_derivative(a, b):
        """z'_i(b) = 1 - expit(a + b)"""
        # b_safe = np.clip(b, -10, 10)
        
        z_prime = 1 - expit((a + b))
        z_prime = np.clip(z_prime, 1e-6, 1 - 1e-6)
        # print("z prime info:")
        # print(np.min(z_prime), np.max(z_prime), np.mean(z_prime))
        return z_prime

    @staticmethod
    def compute_linearization_coefficients(a, b):
        """Compute linearization coefficients"""
        z = CustomLink.evaluate_z(a, b)
        z_prime = CustomLink.evaluate_z_derivative(a, b)
        offset = z - z_prime * b

        return z_prime, offset