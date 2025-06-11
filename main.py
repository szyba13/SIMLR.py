import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.stats import norm

from sklearn.metrics.pairwise import rbf_kernel

def create_sum_matrix(input_list):
    n = len(input_list)
    sum_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sum_matrix[i, j] = input_list[i] + input_list[j]

    return sum_matrix

print("Input matrix:")
input = np.matrix(
[[4.22016824, 9.52059542, 9.06147393, 4.76238876, 8.68718698],
 [3.60131013, 8.88663821, 7.9274555 , 1.77946764, 4.62207242],
 [6.99930763, 6.92595521, 1.5150891 , 2.23769337, 4.32443509],
 [8.59804346, 6.94750078, 9.1130928 , 4.26227738, 7.30300422],
 [3.77543786, 0.43269196, 5.55759746, 2.3849611 , 5.84682024],
 [1.37847937, 6.78821223, 2.67056467, 4.76752098, 3.20457591]])
print(np.matrix(input))
print("-------")

k_value = 3
sig_value = 1

distance = sp.spatial.distance_matrix(input, input)
distance_sorted = distance


print("Distance matrix:")
print(np.matrix(distance))
print("-------")

distance_sorted.sort(axis=0)
distance_sorted = distance_sorted.transpose()

print("Distance matrix:")
print(np.matrix(distance))
print("-------")

mean_distance = distance_sorted[:, 1:k_value+1].sum(axis=1) / k_value

print("Mean distance:")
print(mean_distance)
print("-------")

sig = create_sum_matrix(mean_distance)

sig = np.divide(sig, 2)
sig = np.multiply(sig, sig_value)

print("Variance matrix:")
print(sig)
print("-------")

kernel = np.zeros((6, 6, 6))
kernel = np.asarray(kernel)

for i in range(0, 6):
    kernel[i] = rbf_kernel(np.asarray(input), gamma=(1 / 2 * (sig[i]**2)) )
    kernel[i] = kernel[i]  / ( np.sqrt(2 * np.pi) * sig ) 

print("Kernel matrix:")
print(kernel)
print("-------")


#frobenius norm

# macierze podobieństwa, 
# kernele
# dsytanse
# macierz ograniczenia rzędu

# kenele rbf

