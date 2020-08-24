import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from random import randint
# from copy import deepcopy
# import pandas as pd
# import seaborn as sns

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [(x.strip()).split() for x in content]
    return content

def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def edgeness(img):
    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    absx= cv2.convertScaleAbs(sobel_horizontal)
    absy = cv2.convertScaleAbs(sobel_vertical)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return edge

def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct")
        
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt < x_topleft_p):      
        return 0.0
    if(y_bottomright_gt < y_topleft_p):        
        return 0.0
    if(x_topleft_gt > x_bottomright_p):      
        return 0.0
    if(y_topleft_gt > y_bottomright_p):
        return 0.0
    
    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1 ) * (y_bottomright_p - y_topleft_p + 1)
    
    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area



if __name__ == "__main__":
	### Error + Edgeness + Average Gray ###
	gt_path = glob.glob("./graphs/bdd100k_60_70_GT/*.txt")

	result_file_name = "60_70_enhanced_res_6080anchors_yolo_epoch_57_0.1"

	edgeness_values = []
	avg_grays = []
	error = []

	i=0

	for path in gt_path:
	    img_name = get_file_name(path)

	    gt_boxes_list = file_lines_to_list(f"./graphs/bdd100k_60_70_GT/{img_name}.txt")
	    det_boxes_list = file_lines_to_list(f"./graphs/{result_file_name}/{img_name}.txt")
	    
	    gray_scale = cv2.imread(f"../datasets/bdd100k/bdd100k/bdd100k/images/100k/val/{img_name}.jpg", 0)

	    if len(det_boxes_list) > 0:
	        for gt_box in gt_boxes_list:
	            iou_list_i = []
	            for det_box in det_boxes_list:
	                array_gt = [int(gt_box[1]), int(gt_box[2]), int(gt_box[3]), int(gt_box[4])]
	                array_det = [int(det_box[2]), int(det_box[3]), int(det_box[4]), int(det_box[5])]

	                iou = calc_iou(array_gt, array_det)

	                iou_list_i.append(iou)

	            max_iou_i = np.amax(iou_list_i)

	            cropped = gray_scale[int(gt_box[2]):int(gt_box[4]), int(gt_box[1]):int(gt_box[3])]
	            
	            avg_gray = np.average(cropped)
	            
	            gau_img = cv2.GaussianBlur(cropped,(3,3),0)
	            edge = edgeness(gau_img)
	            sum_edge = np.sum(edge)   
	            
	            edgeness_values.append(sum_edge)
	            avg_grays.append(avg_gray)
	            error.append(1-max_iou_i)

	    else:
	        for gt_box in gt_boxes_list:
	            cropped = gray_scale[int(gt_box[2]):int(gt_box[4]), int(gt_box[1]):int(gt_box[3])]
	            
	            avg_gray = np.average(cropped)
	            
	            gau_img = cv2.GaussianBlur(cropped,(3,3),0)
	            edge = edgeness(gau_img)
	            sum_edge = np.sum(edge)   
	            
	            edgeness_values.append(sum_edge)
	            avg_grays.append(avg_gray)
	            error.append(1)
	    i+=1
	    print(i,f"{len(gt_path)}")

	print("Length Data:", len(edgeness_values), len(error))

	data = []
	for i in range(len(edgeness_values)):
	    data.append([edgeness_values[i], avg_grays[i], error[i]])
	    
	index = randint(0, len(gt_path)-1)
	print(data[index])

	data = np.array(data)

	# Split in to Red-Green
	TP = []
	FP = []

	for data_i in data:
	    if data_i[2] <= 0.5: TP.append(data_i)
	    else: FP.append(data_i)
	        
	print(len(TP), len(FP))
	TP = np.array(TP)
	FP = np.array(FP)

	# Creating figure 
	fig = plt.figure(figsize = (10, 7)) 
	ax = plt.axes(projection = "3d") 
	  
	# Creating plot 
	ax.scatter3D(TP[:,0], TP[:,1], TP[:,2], color="green")
	ax.scatter3D(FP[:,0], FP[:,1], FP[:,2], color="red")

	# plt.title("Detection Analysis") 
	ax.set_xlabel('Edgeness', fontweight ='bold')  
	ax.set_ylabel('Average Gray', fontweight ='bold')  
	ax.set_zlabel('Error', fontweight ='bold')  
	  
	# show plot 
	plt.show() 