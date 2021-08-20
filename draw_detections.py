output_folder = "./filtering_from_orig_yolov4/yolov4_resized_input608"
paths = glob.glob(f"./{output_folder}/*.txt")
print(len(paths))

def draw_det_boxes_waymo(output_folder):
    
    index = randint(0, len(paths)-1)
    path = paths[index]
    img_name = get_file_name(path)
    
    img = plt.imread(f"D:/TranLeAnh/datasets/Waymo/dataset/val/{img_name}.jpg")

    gt_boxes = file_lines_to_list(f"D:\TranLeAnh\datasets\Waymo\dataset\waymo_{data_type}_GT//{img_name}.txt")
    det_boxes = file_lines_to_list(f"./{output_folder}/{img_name}.txt")

    print(len(gt_boxes), len(det_boxes))

    dummy = img.copy()

    # scale = 800/1920
    scale = 1

    for obj in gt_boxes:
        cv2.rectangle(dummy, (int(float(obj[1])*scale), int(float(obj[2])*scale)), (int(float(obj[3])*scale), int(float(obj[4])*scale)), (255, 0, 0), thickness = 2)

    for obj in det_boxes:
        if float(obj[1]) >= 0:

            cv2.rectangle(dummy, (int(float(obj[2])), int(float(obj[3]))), (int(float(obj[4])), int(float(obj[5]))), (0, 255, 0), thickness = 2)

    plt.figure(figsize = (15, 10))
    plt.imshow(dummy)
    plt.show()
