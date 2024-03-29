# CREATED ON 2020-07-13
# Author: TRAN LE ANH
# ----------------------- #
#     USEFUL FUNCTIONS    #     
# ----------------------- #

##### IMPORT #####
import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
##################



# READ TXT FILE TO LIST
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [(x.strip()).split() for x in content]
    return content
# ------------------------------------------------------------------------------------------------


# GET FILE NAME FROM PATH
def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


# Method 2:
def path_extractor(path):
    head, tail = os.path.split(path)
    fname, ext = os.path.splitext(tail)
    return head, fname, ext
# ------------------------------------------------------------------------------------------------


# CLUSTERING ACCURACY
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
# ------------------------------------------------------------------------------------------------


# WRITE XML ANNOTATION FILE
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
# ------------------------------------------------------------------------------------------------


# SOBEL EDGE CALCULATION
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
# ------------------------------------------------------------------------------------------------


# Remove array from List
def remove_element(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')
    return edge
# ------------------------------------------------------------------------------------------------


# 6 Non-maximum Suppression 
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
# ------------------------------------------------------------------------------------------------


# Weighted Bounding Box
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
# ------------------------------------------------------------------------------------------------

# Weighted Bounding Box (Faster)
def faster_wbb(all_boxes, thresh=0.5):
    
    box_array = np.array(all_boxes)
    classes_in_img = list(set(box_array[:,5]))

    # output_bboxes = []
    wbbs = []

    for cls in classes_in_img:
        
        cls_mask = (box_array[:,5] == cls)
        boxes = box_array[cls_mask]
        
        while len(boxes) > 0:
            
            start_time_ = time.time()
            
            boxes = np.array(boxes)

            scores = boxes[:,4]
            order = scores.argsort()[::-1]

            anchor = boxes[order[0]]
            remaining = boxes[order[1:]]
            
            ious = bboxes_iou(anchor[0:4], remaining[:, :4])
            selected = remaining[ious >= thresh]
            one_group = np.concatenate(([anchor], selected), axis=0)
                    
            end_time_ = time.time()
            print("--- Time: %s seconds ---" % (end_time_ - start_time_))

            boxes = remaining[ious < thresh]
            average_box = weighted_coors(one_group)
            
            wbbs.append([cls, average_box[4], average_box[0], average_box[1], average_box[2], average_box[3]])
            
    return wbbs
# ------------------------------------------------------------------------------------------------



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
# ------------------------------------------------------------------------------------------------


# Display Plot Title
plt.gca().set_title("image")
# ------------------------------------------------------------------------------------------------


# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans.fit(box_sizes_array)
print(kmeans.cluster_centers_)
# ------------------------------------------------------------------------------------------------


# WAYMO Size process:
    if img_name[name_len-10:name_len-6] == "SIDE" or img_name[name_len-9:name_len-5] == "SIDE": h = 886
    else: h = 1280
    w = 1920
# ------------------------------------------------------------------------------------------------


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
# ------------------------------------------------------------------------------------------------


# Check if a folder exists
if not os.path.isdir(detection_result_dir):
    os.makedirs(detection_result_dir)
# ------------------------------------------------------------------------------------------------

    
### Sort Array
array_x[array_x[:, 1].argsort()[::-1]]    # highest to lowest
array_x[array_x[:, 1].argsort()]          # lowest to highest
# ------------------------------------------------------------------------------------------------


# 1-channel Image to 3-channel Image
img = cv2.imread("D:\\img.jpg")
gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)

img2 = np.zeros_like(img)
img2[:,:,0] = gray
img2[:,:,1] = gray
img2[:,:,2] = gray
# ------------------------------------------------------------------------------------------------


# PyPI Commands
python setup.py sdist bdist_wheel
twine upload dist/* 
