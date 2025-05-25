import numpy as np
import scipy as sp
import pandas
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix

input = np.random.uniform(0, 10, (9, 9))
print(np.matrix(input))

k = 3

distance = sp.spatial.distance_matrix(input, input)
distance.sort(axis=0)
distance = distance.transpose()

mean_distance = distance[:, 1:k+1].sum(axis=1) / k

print(mean_distance)