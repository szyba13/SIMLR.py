import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from scipy.stats import norm

print("Input matrix:")
input = np.random.uniform(0, 10, (6, 5))
print(np.matrix(input))
print("-------")

k_value = 3
sig_value = 1

distance = sp.spatial.distance_matrix(input, input)
distance.sort(axis=0)
distance = distance.transpose()

print("Distance matrix:")
print(np.matrix(distance))
print("-------")

mean_distance = distance[:, 1:k_value+1].sum(axis=1) / k_value

print("Mean distance:")
print(mean_distance)
print("-------")

sig = np.zeros((mean_distance.size, mean_distance.size))
sig[0] = mean_distance


sig = np.add(np.transpose(sig),  sig)
sig = np.divide(sig, 2)
sig = np.multiply(sig, sig_value)

print("Variance matrix:")
print(sig)
print("-------")

W = norm.cdf(distance, 0, sig)

print("W:")
print(W)
print("-------")

Kernel = (W + W.transpose()) / 2

print("Kernel: ")
print(Kernel)
print("------")

Kernel = np.nan_to_num(Kernel)

print("Kernel: ")
print(Kernel)
print("------")

k = 1/np.sqrt(np.diag(Kernel) + 1)

G = np.multiply(Kernel, np.dot(k, k.transpose()))

G1 = np.zeros((np.diag(Kernel).size, np.diag(Kernel).size))
i = 0
for i in range(0, np.diag(Kernel).size):
    G1[i,i] = np.diag(Kernel)[i]
    
print("G1: ")
print(G1)
print("-------")

G2 = G1.transpose()

Kernel_tmp = (G1 + G2 - 2 * G) / 2

Kernel_tmp = Kernel_tmp - np.diag(np.diag(Kernel_tmp))

print("Kernel_tmp: ")
print(Kernel_tmp)
print("--------")