import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
import pandas as pd
import seaborn as sn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI


def create_mu_matrix(k: int, sorted_distances: npt.NDArray) -> npt.NDArray:
    mean_distances = np.mean(sorted_distances[:, 1 : k + 1], axis=1)
    mu_matrix = mean_distances[:, None] + mean_distances[None, :]
    return mu_matrix


def generate_kernel(
    distance_matrix: npt.NDArray, mu_matrix: npt.NDArray, sigma: float
) -> npt.NDArray:
    kernel = (
        -1
        * np.power(distance_matrix, 2)
        / (np.power(sigma, 2) * np.power(mu_matrix, 2) + 1e-10)
    )
    kernel = np.exp(kernel)
    kernel = kernel / (mu_matrix * sigma * np.sqrt(2 * np.pi))

    # kernel = (kernel + kernel.T) / 2
    # kernel = np.maximum(kernel, 0)
    # kernel /= kernel.sum(axis=1, keepdims=True)

    return kernel


def create_kernels(
    k_values: list[int],
    sig_values: list[float],
    distance_matrix: npt.NDArray,
) -> list[npt.NDArray]:
    print("Generating kernels...")
    sorted_distances = np.sort(distance_matrix, axis=1)
    kernels = []
    for k in k_values:
        for sigma in sig_values:
            mu_matrix = create_mu_matrix(k, sorted_distances)
            new_kernel = generate_kernel(distance_matrix, mu_matrix, sigma)
            kernels.append(new_kernel)
    return kernels


def calculate_weighted_kernels(
    kernels: list[npt.NDArray], weights: list[float]
) -> npt.NDArray:
    n = kernels[0].shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(len(kernels)):
        similarity_matrix = similarity_matrix + np.multiply(kernels[i], weights[i])

    # similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    # similarity_matrix = np.maximum(similarity_matrix, 0)
    # similarity_matrix /= similarity_matrix.sum(axis=1, keepdims=True)
    return similarity_matrix


def calculate_first_gamma(distance_matrix: npt.NDArray, k: int):
    sorted_distance_matrix = np.sort(distance_matrix, axis=1)
    N = distance_matrix.shape[0]
    suma = 0.0
    for i in range(N):
        for j in range(1, k):
            a = sorted_distance_matrix[i, k]
            b = sorted_distance_matrix[i, j]
            suma += (a**2) - (b**2)
    result = suma / (2 * N)
    print(f"Starting gamma value: {result}.")
    return result


def calculate_eigengap(n: int, S: npt.NDArray) -> float:
    In = np.identity(S.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(In - S)
    return eigenvalues[n] - eigenvalues[n - 1]


def update_gamma(old_gamma: float, S: npt.NDArray, clusters_amount: int) -> float:
    eigenvalue = calculate_eigengap(clusters_amount, S)
    if eigenvalue > 1e-6:
        return old_gamma * (1 + (0.5 * eigenvalue))  # w oryginale używają *1.5
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
    wK = calculate_weighted_kernels(kernels, weights)
    N = old_S.shape[0]
    ones = np.ones(N)
    In = np.identity(N)

    v = (1 / (2 * beta)) * ((gamma * (L @ L.T)) - wK)  # Znak jest odwrócony
    # u = (In - (ones * ones.T / N)) * v + (ones / N)

    u = [[0] * N] * N
    a = In - ((ones * ones.T) / N)
    b = (1 / N) * ones
    for i in range(N):
        u[i] = np.dot(a, v[i]) + b

    sigmas = []
    for i in range(N):
        sigmas.append(
            sp.optimize.newton(
                S_func,
                np.mean(u[i]),
                args=(u[i],),
            )
            # sp.optimize.root(S_func, np.mean(u[i]), args=(u[i],)).x[0]
        )
    S = np.zeros((N, N))
    for i in range(N):
        S[i] = np.maximum(u[i] - sigmas[i], 0)
    alpha = 0.8
    result = (alpha * S) + ((1 - alpha) * old_S)

    # result = (result + result.T) / 2
    # result = np.maximum(result, 0)
    # result /= result.sum(axis=1, keepdims=True)

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
    # L = preprocessing.MinMaxScaler().fit_transform(In - S)
    # L = (L + L.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(In - S)
    L = eigenvectors[:, 1 : desired_cluster_amount + 1]
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

    # P = (P + P.T) / 2
    # P = np.maximum(P, 0)
    # P /= P.sum(axis=1, keepdims=True)

    return P


def apply_diffusion(S: npt.NDArray, P: npt.NDArray, t: int) -> npt.NDArray:
    N = S.shape[0]
    H = S
    tau = 0.95
    In = np.identity(N)

    H = tau * (H @ P) + (1 - tau) * In
    # for i in range(t):
    #     H_prev = H
    #     if np.linalg.norm(H - H_prev) < 1e-6:
    #         print(f"Converged at iteration {t}")
    #         break

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
    Kw = calculate_weighted_kernels(kernels, w)
    L = optimize_L_matrix(Kw, desired_cluster_amount)
    gamma = calculate_first_gamma(distance_matrix, k)

    old_w = w
    old_S = Kw
    old_L = L

    old_eigengap = np.inf
    for t in range(5):
        S = optimize_S_matrix(kernels, w, L, gamma, old_S)
        L = optimize_L_matrix(S, desired_cluster_amount)
        w = optimize_w_matrix(kernels, S)
        P = calculate_P_matrix(S, distance_matrix, k)
        S = apply_diffusion(S, P, t)

        gamma = update_gamma(gamma, S, desired_cluster_amount)
        eigengap = calculate_eigengap(desired_cluster_amount, S)
        print(f"{t}. Gamma = {gamma}, eigengap = {eigengap}")

        if old_eigengap - eigengap > 1e-6:
            old_eigengap = eigengap
            old_w = w
            old_S = S
            old_L = L
        else:
            print("Smallest Eigengap reached.")
            break

    print(f"Final weights: {w}.")
    return old_S, old_w, old_L


def main():
    digits, labels = datasets.load_digits(return_X_y=True)
    input_amount = 500
    labels = labels[:input_amount]
    input_matrix = digits[:input_amount]
    # input_matrix = preprocessing.MinMaxScaler().fit_transform(input_matrix)

    # według tekstu, wszystkie komórki były przeskalowane przez f(x) = log10(x+1)
    input_matrix = np.log10(input_matrix + 1)
    distance_matrix = sp.spatial.distance_matrix(input_matrix, input_matrix)
    # distance_matrix = preprocessing.MinMaxScaler().fit_transform(distance_matrix)
    # distance_matrix = np.log10(distance_matrix + 1)

    k_values = np.arange(10, 32, 2).tolist()
    sig_values = [1, 1.25, 1.5, 2]
    clusters_amount = 10

    kernels = create_kernels(k_values, sig_values, distance_matrix)
    # for i in range(0, len(kernels)):
    #     plt.title(f"Kernel no. {i}")
    #     plt.imshow(kernels[i], interpolation="nearest", origin="upper")
    #     plt.colorbar()
    #     plt.show()
    similarity, weights, L = optimalization_process(
        kernels, clusters_amount, distance_matrix, k_values[-1]
    )

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    axs[0].imshow(similarity)
    axs[0].set_title("Similarity matrix")

    sort_inds = np.argsort(labels[:input_amount])
    S_sorted = similarity[sort_inds][:, sort_inds]
    axs[1].imshow(S_sorted)
    axs[1].set_title("Similarity matrix sorted by labels")

    plt.show()

    tsne_model = TSNE(n_components=2, random_state=0)
    simlr_tsne_data = tsne_model.fit_transform(similarity)

    simlr_tsne_data = np.vstack((simlr_tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=simlr_tsne_data, columns=("Dim_1", "Dim_2", "label"))

    raw_tsne_data = tsne_model.fit_transform(input_matrix)

    raw_tsne_data = np.vstack((raw_tsne_data.T, labels)).T
    raw_tsne_df = pd.DataFrame(data=raw_tsne_data, columns=("Dim_1", "Dim_2", "label"))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    axs[0].scatter(data=tsne_df, x="Dim_1", y="Dim_2", c="label")
    axs[0].set_title("t-SNE with SIMLR")
    axs[1].scatter(data=raw_tsne_df, x="Dim_1", y="Dim_2", c="label")
    axs[1].set_title("t-SNE without SIMLR")
    plt.show()

    model = KMeans(clusters_amount, random_state=0)
    simlr_labels = model.fit_predict(L)
    raw_labels = model.fit_predict(input_matrix)

    print("SIMLR score: ")
    NMI_score = NMI(labels, simlr_labels)
    print(f"    NMI score: {NMI_score}.")
    ARI_score = ARI(labels, simlr_labels)
    print(f"    ARI score: {ARI_score}.")

    print("Raw score:")
    NMI_score = NMI(labels, raw_labels)
    print(f"    NMI score: {NMI_score}.")
    ARI_score = ARI(labels, raw_labels)
    print(f"    ARI score: {ARI_score}.")


if __name__ == "__main__":
    main()
