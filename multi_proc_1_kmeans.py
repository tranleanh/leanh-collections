import multiprocessing as mp
from multiprocessing import Pool
# print("Number of processors: ", mp.cpu_count())

import time
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


num_samples = 5000


def kmeans_clustering(X):

	time.sleep(10)

	# K-Means Clustering
	kmeans = KMeans(n_clusters=3, init='random')
	kmeans.fit(X)
	centers=kmeans.cluster_centers_
	labels=kmeans.labels_

	return centers, labels


def run_parallel():

	print("Run Parallel...")

	X1, y1 = make_blobs(n_samples=num_samples, n_features = 3, centers=5, cluster_std=0.4, random_state=0)
	print(X1.shape)

	X2, y2 = make_blobs(n_samples=num_samples, n_features = 3, centers=5, cluster_std=0.4, random_state=0)
	print(X1.shape)

	start_time = time.time()

	pool = Pool(processes=2)

	value1 = pool.apply_async(kmeans_clustering, [X1])
	value2 = pool.apply_async(kmeans_clustering, [X2])

	centers1, labels1 = value1.get()
	centers2, labels2 = value2.get()

	print(centers1)
	print(centers2)

	process_time = time.time() - start_time
	print(f"Parallel: {process_time} seconds")


def run_1cpu():

	print("Run on 1 CPU...")

	X1, y1 = make_blobs(n_samples=num_samples, n_features = 3, centers=5, cluster_std=0.4, random_state=0)
	print(X1.shape)

	X2, y2 = make_blobs(n_samples=num_samples, n_features = 3, centers=5, cluster_std=0.4, random_state=0)
	print(X1.shape)

	start_time = time.time()

	centers1, labels1 = kmeans_clustering(X1)
	centers2, labels2 = kmeans_clustering(X2)

	print(centers1)
	print(centers2)

	process_time = time.time() - start_time
	print(f"1 CPU: {process_time} seconds")


def main():
	run_parallel()
	run_1cpu()


if __name__ == "__main__":
	main()
