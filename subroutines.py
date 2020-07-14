# CREATED ON 2020-07-13
# Author: TRAN LE ANH
# ----------------------- #
#     USEFUL FUNCTIONS    #     
# ----------------------- #

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











