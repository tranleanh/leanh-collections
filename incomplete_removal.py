import numpy as np


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


def incomplete_removal_postprocessing(all_boxes, patch_ranges, iou_thr):
	'''
	This function is to remove all incomplete detection from every patch
	all_boxes: [[object_name, score, x1, y1, x2, y2], [],..., []]
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