#================================================================
#
#   Editor       : Sublime Text
#   Application  : Generate Results with Different Confidence Scores
#   Created Date : 2020-11-03
#   Author       : tranleanh
#   Description  : Shared
#   Version      : 1.0
#
#================================================================

import os
import glob


# Detection Folder Path
detection_path = "./detections/detection_samples"

# Configurations
target_conf_scr = 0.5



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



# MAIN
if __name__ == "__main__":

	txt_src = glob.glob(f"{detection_path}/*.txt")
	total_num_items = len(txt_src)

	directory = f"{detection_path}_cfscr_{target_conf_scr}"
	if not os.path.exists(directory):
	    os.makedirs(directory)

	cnt=0
	for txt_path in txt_src:
		contents = file_lines_to_list(txt_path)

		all_boxes = []
		for content in contents:
			if float(content[1]) >= target_conf_scr: 
				all_boxes.append([content[0], float(content[1]), 
					int(float(content[2])), 
					int(float(content[3])), 
					int(float(content[4])), 
					int(float(content[5]))])

		# Write TXT File
		txt_file = open(f"{directory}/{get_file_name(txt_path)}.txt", "w")
		for obj in all_boxes:
			print(obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], file=txt_file)
		txt_file.close()

		cnt+=1
		print(f"Processed {cnt} out of {total_num_items} items...")