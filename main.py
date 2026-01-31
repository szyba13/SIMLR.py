import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.datasets import load_digits
from sklearn import preprocessing


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

    kernel = (kernel + kernel.T) / 2
    kernel = np.maximum(kernel, 0)
    kernel /= kernel.sum(axis=1, keepdims=True)

    return kernel


def create_kernels(
    input_matrix: npt.NDArray,
    k_values: list[int],
    sig_values: list[float],
    distance_matrix: npt.NDArray,
) -> list[npt.NDArray]:
    print("Generating kernels...")
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


def calculate_first_gamma(distance_matrix: npt.NDArray, k: int):
    sorted_distance_matrix = np.sort(distance_matrix, axis=1)
    N = distance_matrix.shape[0]
    suma = 0.0
    for i in range(1, N):
        for j in range(k + 1):
            a = sorted_distance_matrix[i, k + 1]
            b = sorted_distance_matrix[i, j]
            suma += (a**2) - (b**2)
    result = suma / (N * N)  # N^2 zamiast 2N
    print(f"Starting gamma value: {result}.")
    return result


def calculate_eigengap(n: int, S: npt.NDArray) -> float:
    In = np.identity(S.shape[0])
    L = preprocessing.MinMaxScaler().fit_transform(S)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues[n + 1] - eigenvalues[n]


def update_gamma(old_gamma: float, S: npt.NDArray, clusters_amount: int) -> float:
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
    old_S: npt.NDArray,
) -> npt.NDArray:
    beta = gamma
    S = calculate_similarity(kernels, weights)
    N = S.shape[0]
    ones = np.ones(N)
    In = np.identity(N)

    v = (-1 / (2 * beta)) * ((gamma * (L @ L.T)) - S)
    u = np.zeros((N, N))

    for i in range(N):
        u[i] = np.dot((In - (ones * ones.T / N)), v[i]) + (ones / N)

    sigmas = []
    for i in range(N):
        sigmas.append(
            # sp.optimize.newton(
            #     S_func,
            #     np.mean(u[i]),
            #     args=(u[i],),
            # )
            sp.optimize.root(S_func, np.mean(u[i]), args=(u[i][:-1],)).x[0]
        )

    S = np.maximum(u - sigmas, 0)
    alpha = 0.8
    result = ((1 - alpha) * S) + (alpha * old_S)

    result = (result + result.T) / 2
    result = np.maximum(result, 0)
    result /= result.sum(axis=1, keepdims=True)

    return result


def optimize_w_matrix(kernels: list[npt.NDArray], S: npt.NDArray) -> list[float]:
    po = kernels[0].shape[0]
    exponents = []
    for k in kernels:
        a = np.sum(k * S) / po
        exponents.append(np.exp(a))

    exponents_sum = np.sum(exponents)
    w = []
    for exponent in exponents:
        w.append(exponent / exponents_sum)
    return w


def optimize_L_matrix(S: npt.NDArray, desired_cluster_amount: int) -> npt.NDArray:
    In = np.identity(S.shape[0])
    L = preprocessing.MinMaxScaler().fit_transform(In - S)
    L = (L + L.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    L = eigenvectors[:, :desired_cluster_amount]
    return np.array(L)


def calculate_P_matrix(
    S: npt.NDArray, distance_matrix: npt.NDArray, k: int
) -> npt.NDArray:
    top_indices = np.argsort(distance_matrix, axis=1)[:, 1 : k + 1]
    N = S.shape[0]
    P = np.zeros((N, N))
    for i in range(N):
        suma = np.sum(S[i][top_indices[i]], axis=0)
        if suma > 0:
            for index in top_indices[i]:
                P[i][index] = S[i][index] / suma

    P = (P + P.T) / 2
    P = np.maximum(P, 0)
    P /= P.sum(axis=1, keepdims=True)

    return P


def apply_diffusion(S: npt.NDArray, P: npt.NDArray) -> npt.NDArray:
    N = S.shape[0]
    H = S
    tau = 1 - (1 / N)  # tau = 0.8
    In = np.identity(N)
    H = tau * (H @ P) + (1 - tau) * In

    H = (H + H.T) / 2
    H = np.maximum(H, 0)
    H /= H.sum(axis=1, keepdims=True)

    return H


def optimalization_process(
    kernels: list[npt.NDArray],
    desired_cluster_amount: int,
    distance_matrix: npt.NDArray,
    k: int,
):
    print("Starting optimalization process...")
    w = [1 / len(kernels)] * len(kernels)
    S = calculate_similarity(kernels, w)
    L = optimize_L_matrix(S, desired_cluster_amount)
    gamma = calculate_first_gamma(distance_matrix, k)

    old_w = w
    old_S = S

    old_eigengap = np.inf
    for t in range(20):
        S = optimize_S_matrix(kernels, w, L, gamma, S)
        L = optimize_L_matrix(S, desired_cluster_amount)
        w = optimize_w_matrix(kernels, S)
        P = calculate_P_matrix(S, distance_matrix, k)
        S = apply_diffusion(S, P)

        gamma = update_gamma(gamma, S, desired_cluster_amount)
        eigengap = calculate_eigengap(desired_cluster_amount, S)
        print(f"{t}. Gamma = {gamma}, eigengap = {eigengap}")

        if old_eigengap > eigengap or eigengap > 0.1:
            old_eigengap = eigengap
            old_w = w
            old_S = S
        else:
            print("Smallest Eigengap reached.")
            break

    print(f"Final weights: {w}.")
    return old_S, old_w


def main():
    digits, labels = load_digits(return_X_y=True)
    input_amount = 500
    input_matrix = digits[:input_amount]
    input_matrix = preprocessing.MinMaxScaler().fit_transform(input_matrix)
    distance_matrix = sp.spatial.distance_matrix(input_matrix, input_matrix)

    k_values = np.arange(10, 31, 2).tolist()
    sig_values = [1, 1.25, 1.5, 2]
    clusters_amount = 10

    kernels = create_kernels(input_matrix, k_values, sig_values, distance_matrix)
    similarity, weights = optimalization_process(
        kernels, clusters_amount, distance_matrix, k_values[-1]
    )

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.title("Optimized Similarity Matrix")
    plt.imshow(similarity, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()

    # Sort S_final by ground truth labels to see block diagonal structure
    sort_inds = np.argsort(labels[:input_amount])
    S_sorted = similarity[sort_inds][:, sort_inds]

    plt.figure(figsize=(10, 8))
    plt.imshow(S_sorted, interpolation="nearest", origin="upper", cmap="viridis")
    plt.colorbar(label="Similarity Probability")
    plt.title("Optimized Similarity Matrix (Sorted by Label)")
    plt.show()


if __name__ == "__main__":
    main()
