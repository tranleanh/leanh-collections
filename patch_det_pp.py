#================================================================
#
#   Editor       : Sublime Text
#   Application  : Post-processing for Sub-patch Detection
#   Created Date : 2020-11-02
#   Description  : Shared
#   Version      : 1.0
#
#================================================================

import os
import numpy as np
import glob


# Detection Folder Path
detection_path = "subsampling_608_304_epoch_49_no_margin_update1"

# Configurations
patch_ranges = [[0, 800], [560, 1360], [1120, 1920]]
partly_iou_thr = 0.5
wwb_iou_thr = 0.5



def file_lines_to_list(path):
	'''
	This function is to read txt file lines as a list
	'''
	with open(path) as f:
		content = f.readlines()
	content = [(x.strip()).split() for x in content]
	return content


def get_file_name(path):
	'''
	This function is to get file name from a path
	'''
	basename = os.path.basename(path)
	onlyname = os.path.splitext(basename)[0]
	return onlyname


def calc_iou(bbox1, bbox2):
	'''
	This function is to calculate IoU between 2 boxes
	bbox = [x1, y1, x2, y2]

	'''
	x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = bbox1
	x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = bbox2
	
	if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
		raise AssertionError("Ground Truth Bounding Box is not correct")
	if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
		raise AssertionError("Predicted Bounding Box is not correct")
		
	# If bbox1 and bbox2 do not overlap then IoU = 0
	if(x_bottomright_gt < x_topleft_p):      
		return 0.0
	if(y_bottomright_gt < y_topleft_p):        
		return 0.0
	if(x_topleft_gt > x_bottomright_p):      
		return 0.0
	if(y_topleft_gt > y_bottomright_p):
		return 0.0
	
	bbox1_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
	bbox2_area = (x_bottomright_p - x_topleft_p + 1 ) * (y_bottomright_p - y_topleft_p + 1)
	
	x_top_left = np.max([x_topleft_gt, x_topleft_p])
	y_top_left = np.max([y_topleft_gt, y_topleft_p])
	x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
	y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
	
	intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left  + 1)
	union_area = (bbox1_area + bbox2_area - intersection_area)
   
	return intersection_area/union_area


def incomplete_removal(all_boxes, patch_ranges, iou_thr):
	'''
	This function is to remove all incomplete detection from every patch
	all_boxes: [[object_name, score, x1, y1, x2, y2], [],..., []]
	patch_ranges: [[x_min, x_max],[],...,[]]
	'''
	for x_range in patch_ranges:
		patch_boxes = []

		for box in all_boxes:
			x1 = int(box[2])
			y1 = int(box[3])
			x2 = int(box[4])
			y2 = int(box[5])

			if x1 < x_range[1] and x2 > x_range[0]:
				patch_boxes.append([box[0], box[1], x1, y1, x2, y2])

		inside_boxes = []
		outside_boxes = []

		for box in patch_boxes:
			x1 = int(box[2])
			y1 = int(box[3])
			x2 = int(box[4])
			y2 = int(box[5])

			if x1 >= x_range[0] and x2 <= x_range[1]:
				inside_boxes.append([box[0], box[1], x1, y1, x2, y2])
			else:
				if x1 < x_range[0]:
					x1_inside_part = x_range[0]+1
				else: x1_inside_part = x1

				if x2 > x_range[1]:
					x2_inside_part = x_range[1]-1
				else: x2_inside_part = x2 

				inside_part = [box[0], box[1], x1_inside_part, y1, x2_inside_part, y2]
				full_box = [box[0], box[1], x1, y1, x2, y2]
				outside_boxes.append([inside_part, full_box])

		verified_boxes = []
		error_boxes = []

		for i_box in inside_boxes:
			i_box_coors = [i_box[2], i_box[3], i_box[4], i_box[5]]
			error_box = 0

			for o_box_pair in outside_boxes:
				o_box_coors = [o_box_pair[0][2], o_box_pair[0][3], o_box_pair[0][4], o_box_pair[0][5]]

				iou = calc_iou(i_box_coors, o_box_coors)

				if iou > iou_thr: 
					error_box+=1
					error_boxes.append(i_box)
					break

			if error_box == 0:
				verified_boxes.append(i_box)

		new_all_boxes = []
		for box in all_boxes:
			if box not in error_boxes: new_all_boxes.append(box)

		all_boxes = new_all_boxes

	corrected_boxes = all_boxes

	return corrected_boxes


def weighted_coors(boxes):
	'''
	boxes = [x1, y1, x2, y2, score]
	'''
	boxes = np.array(boxes)
	max_score = np.amax(boxes[:,4])
	sum_score = sum(boxes[:,4])
	out_x1 = sum(boxes[:,0]*boxes[:,4])/sum_score
	out_y1 = sum(boxes[:,1]*boxes[:,4])/sum_score
	out_x2 = sum(boxes[:,2]*boxes[:,4])/sum_score
	out_y2 = sum(boxes[:,3]*boxes[:,4])/sum_score

	return [int(out_x1), int(out_y1), int(out_x2), int(out_y2), max_score]


def wbb(all_boxes, thresh=0.5):
	'''
	This function is to apply Weighted Bounding Box
	all_boxes: [[object_name, score, x1, y1, x2, y2], [],..., []]
	'''
	box_array = np.array(all_boxes)
	classes_in_img = list(set(box_array[:,0]))

	output_bboxes = []

	for cls in classes_in_img:
		cls_mask = (box_array[:,0] == cls)
		cls_bboxes = box_array[cls_mask]

		boxes = []
		for box in cls_bboxes:
			boxes.append([int(box[2]), int(box[3]), int(box[4]), int(box[5]), float(box[1])])

		boxes = np.array(boxes)
		wbbs = []

		while len(boxes) > 0:
			boxes = np.array(boxes)

			scores = boxes[:,4]
			order = scores.argsort()[::-1]

			anchor = boxes[order[0]]
			one_group = []
			one_group.append(anchor)

			new_boxes = []

			for i in range(len(order)-1):
				test_box = boxes[order[i+1]]
				iou = calc_iou(anchor[0:4], test_box[0:4])
				if iou >= thresh: one_group.append(test_box)
				else: new_boxes.append(test_box)

			boxes = new_boxes    
			average_box = weighted_coors(one_group)
			wbbs.append(average_box)

		for box in wbbs: output_bboxes.append([cls, box[4], box[0], box[1], box[2], box[3]])
		
	return output_bboxes


# MAIN
if __name__ == "__main__":

	txt_src = glob.glob(f"{detection_path}/*.txt")

	directory = f"{detection_path}_processed"
	if not os.path.exists(directory):
	    os.makedirs(directory)

	cnt=0
	for txt_path in txt_src:
		contents = file_lines_to_list(txt_path)

		all_boxes = []
		for content in contents:
			all_boxes.append([content[0], float(content[1]), 
				int(float(content[2])), 
				int(float(content[3])), 
				int(float(content[4])), 
				int(float(content[5]))])

		# Incomplete Removal
		corrected_boxes = incomplete_removal(all_boxes, patch_ranges, partly_iou_thr)

		# Weighted Bounding Box
		wbb_boxes = wbb(corrected_boxes, wwb_iou_thr)

		# Write TXT File
		txt_file = open(f"{directory}/{get_file_name(txt_path)}.txt", "w")
		for obj in wbb_boxes:
			print(obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], file=txt_file)
		txt_file.close()

		cnt+=1
		print(f"Processed {cnt} out of {len(txt_src)} items...")