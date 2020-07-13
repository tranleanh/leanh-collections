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