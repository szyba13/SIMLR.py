from math import dist
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.datasets import load_digits


def create_epsilon_matrix(
    k: int, sigma: float, sorted_distances: npt.NDArray
) -> npt.NDArray:
    mean_distances = np.mean(sorted_distances[:, 1 : k + 1], axis=1)
    epsilon_matrix = mean_distances[:, None] + mean_distances[None, :]
    epsilon_matrix = np.divide(epsilon_matrix, 2)
    epsilon_matrix = np.multiply(epsilon_matrix, sigma)
    return epsilon_matrix


def generate_kernel(
    distance_matrix: npt.NDArray, epsilon_matrix: npt.NDArray
) -> npt.NDArray:
    kernel = np.power(distance_matrix, 2) / (-2 * np.power(epsilon_matrix, 2))
    kernel = np.exp(kernel)
    kernel = kernel / (epsilon_matrix * np.sqrt(2 * np.pi))
    return kernel


def create_kernels(
    input_matrix: npt.NDArray,
    k_values: list[int],
    sig_values: list[float],
    distance_matrix: npt.NDArray,
) -> list[npt.NDArray]:
    sorted_distances = np.sort(distance_matrix, axis=1)
    kernels = []
    for k in k_values:
        for sigma in sig_values:
            epsilon_matrix = create_epsilon_matrix(k, sigma, sorted_distances)
            new_kernel = generate_kernel(distance_matrix, epsilon_matrix)
            kernels.append(new_kernel)
    return kernels


def calculate_similarity(
    kernels: list[npt.NDArray], weights: list[float]
) -> npt.NDArray:
    n = kernels[0].shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(len(kernels)):
        similarity_matrix = similarity_matrix + np.multiply(kernels[i], weights[i])
    return similarity_matrix


def calculate_first_gamma(distance_matrix: npt.NDArray, k_values: list[int]):
    distance_matrix = np.sort(distance_matrix, axis=1)
    N = distance_matrix.shape[0]
    suma = 0.0
    for i in range(1, N):
        for j in range(k_values[-1]):
            a = distance_matrix[i, k_values[-1] + 1]
            b = distance_matrix[i, j]
            suma += (a**2) - (b**2)
    return suma / (2 * N)


def calculate_eigengap(n: int, S: npt.NDArray) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    return eigenvalues[n + 1] - eigenvalues[n]


def update_gamma(old_gamma: float, S: npt.NDArray) -> float:
    eigenvalue = calculate_eigengap(clusters_amount, S)
    if eigenvalue > 1e-6:
        return old_gamma * (1 + (0.5 * eigenvalue))
    else:
        return old_gamma


def S_func(sigma: float, u_i: npt.NDArray) -> np.floating:
    return np.mean(np.maximum(sigma - u_i, 0)) - sigma


def optimize_S_matrix(
    kernels: list[npt.NDArray],
    weights: list[float],
    L: npt.NDArray,
    gamma: float,
    S: npt.NDArray,
) -> npt.NDArray:
    beta = gamma
    # S = calculate_similarity(kernels, weights)
    N = S.shape[0]
    ones = np.ones(N)
    In = np.identity(N)
    v = (-1 / (2 * beta)) * (gamma * (L @ L.T) - S)
    u = (In - (ones * ones.T / N)) @ v + (ones / N)

    sigmas = []
    for i in range(N):
        sigmas.append(
            sp.optimize.newton(
                S_func,
                np.mean(u[i]),
                args=(u[i],),
            )
            # sp.optimize.root(S_func, np.mean(u[i]), args=(u[i],))
        )

    S = np.maximum(u - sigmas, 0)
    return S


def optimize_w_matrix(kernels: list[npt.NDArray], S: npt.NDArray) -> list[float]:
    po = kernels[0].shape[0]
    exponents = []
    for k in kernels:
        a = np.sum(k @ S) / po
        exponents.append(np.exp(a))

    exponents_sum = np.sum(exponents)
    w = []
    for exponent in exponents:
        w.append(exponent / exponents_sum)
    return w


def optimize_L_matrix(S: npt.NDArray) -> npt.NDArray:
    In = np.identity(S.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh((In - S))
    index_order = np.argsort(eigenvalues)
    top_indices = index_order[:clusters_amount]
    L = [eigenvectors[:, i] for i in top_indices]
    return np.array(L).T


def diffusion(S: npt.NDArray, t: int, distance_matrix: npt.NDArray) -> npt.NDArray:
    distance_matrix = np.argsort(distance_matrix, axis=1)
    top_indices = distance_matrix[:, 1 : k_values[-1]]
    N = S.shape[0]
    mask = np.zeros((N, N))
    for i, line in enumerate(top_indices):
        for j in line:
            mask[i][j] = 1

    # suma = np.sum(np.multiply(S, mask), axis=0)
    # P = np.zeros((N, N))
    # P = np.multiply(np.divide(S.T, suma).T, mask)

    P = (S / np.sum(S, axis=0)) * mask

    H = S
    tau = 0.8
    In = np.identity(N)
    for i in range(t):
        H = tau * (H @ P) + (1 - tau) * In

    return H


def optimalization_process(
    kernels: list[npt.NDArray],
    desired_cluster_amount: int,
    distance_matrix: npt.NDArray,
):
    w = [1 / len(kernels)] * len(kernels)
    S = calculate_similarity(kernels, w)
    L = optimize_L_matrix(S)
    gamma = calculate_first_gamma(distance_matrix, k_values)

    old_eigengap = np.inf
    for t in range(20):
        S = optimize_S_matrix(kernels, w, L, gamma, S)
        L = optimize_L_matrix(S)
        w = optimize_w_matrix(kernels, S)
        gamma = update_gamma(gamma, S)
        S = diffusion(S, t, distance_matrix)
        eigengap = calculate_eigengap(desired_cluster_amount, S)
        if old_eigengap > eigengap:
            old_eigengap = eigengap
        else:
            break

    return w


if __name__ == "__main__":
    digits, labels = load_digits(return_X_y=True)
    input_matrix = digits[:200]
    distance_matrix = sp.spatial.distance_matrix(input_matrix, input_matrix)

    k_values = [5, 6, 7]
    sig_values = [1, 1.25, 1.5, 2]
    clusters_amount = 10

    kernels = create_kernels(input_matrix, k_values, sig_values, distance_matrix)
    weights = optimalization_process(kernels, clusters_amount, distance_matrix)
    print()

    similarity = calculate_similarity(kernels, weights)
    plt.imshow(similarity, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()
