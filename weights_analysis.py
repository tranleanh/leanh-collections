import os
import cv2
import numpy as np
import random


def get_file_name(path):
	basename = os.path.basename(path)
	onlyname = os.path.splitext(basename)[0]
	return onlyname


def read_darknet_model(config_file, weight_file):
	return cv2.dnn.readNetFromDarknet(config_file, weight_file)


def read_layer_names(net):
	return net.getLayerNames()


def read_layer_filter(net, layer_name):
	return net.getParam(layer_name)


def save_conv_filters(net, save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	ln = net.getLayerNames()
	for layer_name in ln:
		if layer_name[0:4] == "conv":
			layer_filter = net.getParam(layer_name)
			np.save(f"{save_path}/{layer_name}.npy", layer_filter)
	print("Weight saved!")


def save_conv_filters_16bit(net, save_path):
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	ln = net.getLayerNames()
	for layer_name in ln:
		if layer_name[0:4] == "conv":
			layer_filter = net.getParam(layer_name)
			layer_filter16 = np.array(layer_filter).astype(np.float16)
			np.save(f"{save_path}/{layer_name}.npy", layer_filter16)
	print("Weight converted from 32bit to 16 bit!")


def read_saved_filter(file_path):
	layername = get_file_name(file_path)
	layer_filter = np.load(file_path)
	return layername, layer_filter.shape, layer_filter



if __name__ == "__main__":

	config_file = "yolov4-waymo100k_800.cfg"
	weight_file = "yolov4-waymo100k_800_best.weights"

	net = read_darknet_model(config_file, weight_file)
	layer_names = read_layer_names(net)
	print(layer_names)


	# Save Filters
	save_path = "saved_filters"
	save_conv_filters(net, save_path)


	# Save Filters 16 bit
	save_path_16bit = "saved_filters_16bit"
	save_conv_filters_16bit(net, save_path_16bit)


	# Read Filter Weights
	# conv_name = random.sample(layer_names, 1)
	conv_name = "conv_0"
	filter_32bit = read_saved_filter(f"{save_path}/{conv_name}.npy")
	filter_16bit = read_saved_filter(f"{save_path_16bit}/{conv_name}.npy")

	print(filter_32bit)
	print(filter_16bit)