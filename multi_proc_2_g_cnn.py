import multiprocessing as mp
from multiprocessing import Pool
# print("Number of processors: ", mp.cpu_count())

import numpy as np
import time
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

from g_centroid_neural_networks import remove_element, centroid_neural_network, centroid_neural_network_detected_weights
from gcnn_subroutines import centroid_neural_network_detected_weights, g_centroid_neural_network


def run_centroid_nn(X):

	# Centroid Neural Net
	num_clusters = 10
	# start = time.perf_counter()
	centroids, w, cluster_indices, cluster_elements = centroid_neural_network(X, num_clusters, max_iteration = 1000, epsilon = 0.05)
	# end = time.perf_counter()
	# process_time = end - start
	# print("Processing Time:", process_time)

	return centroids, w, cluster_indices, cluster_elements


def run_parallel():

	print("Run Parallel...")

	X, y = make_blobs(n_samples=5000, centers=10, cluster_std=0.9, random_state=0)
	# X, y = make_blobs(n_samples=1200, centers=3, cluster_std=0.9, random_state=0)
	print(X.shape)

	start_time = time.time()

	num_subdata = 4
	new_data = []
	for i in range(num_subdata):
		subdata = []
		for j in range(len(X)//num_subdata):
			x_i = X[(len(X)//num_subdata)*i + j]
			subdata.append(x_i)
		new_data.append(subdata)
	new_data = np.array(new_data)
	# print(np.array(new_data).shape)

	subdata_1 = new_data[0]
	subdata_2 = new_data[1]
	subdata_3 = new_data[2]
	subdata_4 = new_data[3]

	pool = Pool(processes=num_subdata)

	value1 = pool.apply_async(run_centroid_nn, [subdata_1])
	value2 = pool.apply_async(run_centroid_nn, [subdata_2])
	value3 = pool.apply_async(run_centroid_nn, [subdata_3])
	value4 = pool.apply_async(run_centroid_nn, [subdata_4])

	centroids1, w1, cluster_indices1, cluster_elements1 = value1.get()
	centroids2, w2, cluster_indices2, cluster_elements2 = value2.get()
	centroids3, w3, cluster_indices3, cluster_elements3 = value3.get()
	centroids4, w4, cluster_indices4, cluster_elements4 = value4.get()

	# Create New Data with Detected Centroids
	gen2_data = []

	for centroids_i in centroids1:
		gen2_data.append(centroids_i)

	for centroids_i in centroids2:
		gen2_data.append(centroids_i)

	for centroids_i in centroids3:
		gen2_data.append(centroids_i)

	for centroids_i in centroids4:
		gen2_data.append(centroids_i)

	gen2_data = np.array(gen2_data)

	max_iteration=1000
	epsilon=0.05
	num_clusters = 10

	# Run G-CNN one more time
	centroids_2, w_2, cluster_indices_2, cluster_elements_2 = centroid_neural_network(gen2_data, num_clusters, max_iteration, epsilon)

	# Run G-CNN last time
	detected_weights = centroids_2
	centroids, weights, cluster_indices, cluster_elements = centroid_neural_network_detected_weights(X, detected_weights, num_clusters, max_iteration)
	print("Reach the Desired Number of Clusters. Stop!")

	print(centroids)
	# print(centers2)

	process_time = time.time() - start_time
	print(f"Parallel: {process_time} seconds")


def run_1cpu():

	print("Run on 1 CPU...")

	X, y = make_blobs(n_samples=5000, centers=10, cluster_std=0.9, random_state=0)
	# X, y = make_blobs(n_samples=1200, centers=3, cluster_std=0.9, random_state=0)
	print(X.shape)

	start_time = time.time()

	centroids, w, cluster_indices, cluster_elements = run_centroid_nn(X)

	print(centroids)

	process_time = time.time() - start_time
	print(f"1 CPU: {process_time} seconds")


def main():

	run_parallel()
	# run_1cpu()


if __name__ == "__main__":
	main()
