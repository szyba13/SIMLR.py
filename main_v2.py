import enum
import numpy as np
import numpy.typing as npt
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class SIMLRsolver:

    def __init__(self, input_matrix: npt.ArrayLike) -> None:
        self.input_matrix: npt.NDArray[np.float64] = np.array(input_matrix)
        self.input_size: int = max(self.input_matrix.shape)
        self.output_shape: tuple[int, int] = (self.input_size, self.input_size)
        self.distance_matrix: npt.NDArray[np.float64] = sp.spatial.distance_matrix(
            input_matrix, input_matrix
        )
        self.distance_matrix_sorted = np.sort(self.distance_matrix, axis=1)


    def set_params(self, k: list[int], sig: list[float], C: int) -> None:
        self.k_values = k
        self.sig_values = sig
        self.clusters_amount = C
        self.kernels_amount: int = len(self.k_values) * len(self.sig_values)
        self.weights = [1 / self.kernels_amount] * self.kernels_amount


    def pairwise_addition(self, input_matrix: npt.NDArray) -> npt.NDArray:
        result = np.zeros(self.output_shape)
        for i in range(self.input_size):
            for j in range(self.input_size):
                result[i][j] = input_matrix[i] + input_matrix[j]
        return result


    def create_epsilon_matrix(self, k: int, sigma: float) -> npt.NDArray:
        mean_distances = self.distance_matrix_sorted[:, 1:k+1].sum(axis=1) / k

        epsilon_matrix: npt.NDArray[np.float64] = self.pairwise_addition(mean_distances)
        epsilon_matrix = np.divide(epsilon_matrix, 2)
        epsilon_matrix = np.multiply(epsilon_matrix, sigma)
        return epsilon_matrix


    def generate_kernel(self, epsilon_matrix: npt.NDArray) -> npt.NDArray:
        kernel: npt.NDArray[np.float64] = np.zeros(self.output_shape)
        kernel = np.asarray(kernel)

        kernel = np.pow(self.distance_matrix, 2) / (-2 * np.pow(epsilon_matrix, 2))
        kernel = np.exp(kernel)
        kernel = kernel / (epsilon_matrix * np.sqrt(2 * np.pi))

        return kernel


    def create_kernels(self) -> list[npt.NDArray]:
        self.kernels: list[npt.NDArray] = []
        for k in self.k_values:
            for sigma in self.sig_values:
                epsilon_matrix = self.create_epsilon_matrix(k, sigma)
                new_kernel = self.generate_kernel(epsilon_matrix)
                self.kernels.append(new_kernel)
        self.calculate_similarity()
        return self.kernels


    def calculate_similarity(self) -> None:
        similarity = np.zeros(self.output_shape)
        for i in range(self.kernels_amount):
            similarity = similarity + np.multiply(self.kernels[i], self.weights[i])
        self.similarity = similarity


    def calculate_eigengap(self) -> float:
        In = np.identity(self.input_size)
        eigenvalues, eigenvectors = np.linalg.eig(self.similarity)
        return eigenvalues[self.clusters_amount] - eigenvalues[self.clusters_amount - 1]


    def calculate_gamma(self) -> float:
        suma = 0.0
        for i in range(1, self.input_size):
            for j in range(self.k_values[-1]):
                a = self.distance_matrix_sorted[i, self.k_values[-1]+1]
                b = self.distance_matrix_sorted[i, j]
                suma += (a**2) - (b**2)

        return suma / (2 * self.input_size)


    def update_gamma(self, old_gamma: float) -> float:
        eigengap = self.calculate_eigengap()
        if eigengap > 1e-6:
            return old_gamma * (1 + (0.5 * eigengap))
        else:
            return old_gamma


    def optimize_laplacian_matrix(self):
        In = np.identity(self.input_size)

        eigenvalues, eigenvectors = np.linalg.eig((In - self.similarity))
        index_order = np.argsort(eigenvalues)
        top_indices = index_order[: self.clusters_amount]
        L = [eigenvectors[:, i] for i in top_indices]
        self.laplacian = np.array(L).T


    def optimize_similarity_matrix(self, gamma):
        beta = gamma
        ones = np.ones(self.input_size)
        In = np.identity(self.input_size)

        va = -1 / (2 * beta)
        vb = gamma * (self.laplacian @ self.laplacian.T) - self.similarity
        v = va * vb

        ua = (In - (ones @ ones.T / self.input_size)) * v
        ub = ones / self.input_size
        u = ua + ub

        def S_func(sigma, u_i):
            suma = 0
            for j in range(1, self.input_size - 1):
                suma += np.maximum(sigma - u_i[j], 0)
            return suma / (self.input_size - 1) - sigma

        sigmas = []
        for i in range(self.input_size):
            sigmas.append(sp.optimize.newton(S_func, np.mean(u[i]), args=(u[i],)))

        self.similarity = np.maximum(np.subtract(u.T, sigmas).T, 0)


    def optimize_weights_matrix(self):
        po = 0.1
        exponents = []
        for i in range(self.kernels_amount):
            a = np.sum(self.kernels[i] * self.similarity) / po
            exponents.append(np.exp(a))

        exponents_sum = np.sum(exponents)
        w = []
        for i in range(self.kernels_amount):
            w.append(exponents[i] / exponents_sum)

        self.weights = w


    def diffusion(self, t: int):
        distance_indices = np.argsort(self.distance_matrix, axis=1)
        top_indices = distance_indices[:, 1:self.k_values[0]]
        mask = np.zeros(self.output_shape)
        for i, line in enumerate(top_indices):
            for j in line:
                mask[i][j] = 1

        suma = np.sum(np.multiply(self.similarity, mask), axis=0)
        P = np.zeros(self.output_shape)
        P = np.multiply(np.divide(self.similarity.T, suma).T, mask)

        H = self.similarity
        tau = 0.8
        In = np.identity(self.input_size)
        for i in range(t):
            H = tau * H * P + (1 - tau) * In

        self.similarity = H


    def optimalization_process(self) -> npt.NDArray:
        self.optimize_laplacian_matrix()
        gamma = self.calculate_gamma()

        t = 0
        old_eigengap = 10
        for i in range(100):
            self.optimize_similarity_matrix(gamma)
            self.optimize_laplacian_matrix()
            self.optimize_weights_matrix()
            gamma = self.update_gamma(gamma)
            self.diffusion(t)

            eigengap = self.calculate_eigengap()
            if old_eigengap > eigengap:
                old_eigengap = eigengap
            else:
                break
        return self.similarity


if __name__ == "__main__":
    digits = load_digits()
    input_matrix = digits.data[:100]

    k_values = [5, 6, 7]
    sig_values = [1, 1.25, 1.5, 2]
    clusters_amount = 10
    
    s = SIMLRsolver(input_matrix)
    s.set_params(k_values, sig_values, clusters_amount)
    s.create_kernels()
    similarity = s.optimalization_process()

    plt.imshow(similarity, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()

    print()
