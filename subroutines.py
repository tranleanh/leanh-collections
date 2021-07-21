# CREATED ON 2020-07-13
# Author: TRAN LE ANH
# ----------------------- #
#     USEFUL FUNCTIONS    #     
# ----------------------- #

##### IMPORT #####
import os
import glob
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
##################


# 1. READ TXT FILE TO LIST
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [(x.strip()).split() for x in content]
    return content
# ------------------------

# 2. GET FILE NAME FROM PATH
def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname
# ------------------------


# 4. CLUSTERING ACCURACY
def clustering_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    if len(y_true) != len(y_pred):
        raise ValueError("Prediction and Label are not in a same size!!!")
    else:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
        true_cases = 0
        for infor in cm:
            true_cases += np.amax(infor)
    return true_cases / len(y_true)
# ------------------------


# 3. WRITE XML ANNOTATION FILE
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def write_xml_anno(true_labels):
    annotation = Element('annotation')

    for obj in true_labels:
        SubElement(annotation, 'filename').text = no_car_list[rand_back][0] + "_add.jpg"

        object_ = Element('object')
        SubElement(object_, 'name').text = obj[0]

        bndbox = Element('bndbox')
        SubElement(bndbox, 'xmin').text = obj[1]
        SubElement(bndbox, 'ymin').text = obj[2]
        SubElement(bndbox, 'xmax').text = obj[3]
        SubElement(bndbox, 'ymax').text = obj[4]

        object_.append(bndbox)

        annotation.append(object_)

    # Write XML
    with open(osp.join("./addition-xml/" + no_car_list[rand_back][0] + "_" + str(rand_fore[0]) + ".xml"), 'w') as f:
        f.write(prettify(annotation))
        
    return 0
# ------------------------

# 4. SOBEL EDGE CALCULATION
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
# ------------------------

# Write Text
text_file = open("aug_data_names.txt", "w+")
for i in range(len(sub_img_src)//4):
    print(sub_img_names[4*i] + "_stack", file=text_file)
text_file.close()

# 5 Remove array from List
def remove_element(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

# 6 Non-maximum Suppression 
import numpy as np
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# Copy-Paste File
dst_dir = "./waymo/waymo_20per"
i=0
for name in waymo_20per_names:
    filepath = f"../datasets/waymo/data/train/{name}.jpg"
    shutil.copy(filepath, dst_dir)
    i+=1
    print(i, len(waymo_20per_names))

# Weighted Bounding Box
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

def wbb(dets, thresh):
    boxes = dets
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

    return wbbs
    
    return [int(out_x1), int(out_y1), int(out_x2), int(out_y2), max_score]


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


def process_single_image_results(gt_boxes, pred_boxes, iou_thr):

    detected_obj_boxes = []

    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
    
        gt_idx_thr=[]
        pred_idx_thr=[]
        ious=[]

        for igb, gt_box in enumerate(gt_boxes):
            for ipb, pred_box in enumerate(pred_boxes):
                iou = calc_iou(gt_box, pred_box)

                if iou >= iou_thr:
                    detected_obj_boxes.append(gt_box)
    
    return detected_obj_boxes


def sobel_edges(img):  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_16S, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(gray, cv2.CV_16S, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return edge

def sobel_edges_gaublur(img, filter_size = (5,5)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gau_img = cv2.GaussianBlur(gray,filter_size,0)
    x = cv2.Sobel(gau_img, cv2.CV_16S, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(gau_img, cv2.CV_16S, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return edge


# Display Plot Title
plt.gca().set_title("image")


# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans.fit(box_sizes_array)
print(kmeans.cluster_centers_)



# WAYMO Size process:
    if img_name[name_len-10:name_len-6] == "SIDE" or img_name[name_len-9:name_len-5] == "SIDE":
        h = 886
        # print(img_name, h)
        
    else:
        h = 1280
        # print(img_name, h)
        

# Post-process Segmentation
def post_process(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    ret, r_thr = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
    ret, g_thr = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
    ret, b_thr = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
    
    rgbArray = np.zeros(img.shape, 'uint8')
    rgbArray[..., 0] = r_thr
    rgbArray[..., 1] = g_thr
    rgbArray[..., 2] = b_thr
    
    return rgbArray
