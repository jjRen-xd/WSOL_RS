#!/usr/bin/python

# pascol voc to coco
# Convert pascol voc annotation xml to COCO json format.

# pip install lxml
# python voc2coco.py xmllist.txt ../Annotations output.json
# The xmllist.txt is the list of xml file names to convert. 000005.xml 000007.xml 000009.xml

# The "../Annotations" is the place where all xmls located.

# The "output.json" is the output json file.
import re
import argparse
import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}



def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        # 使用正则表达式提取filename中的数字
        imageID = re.findall(r'\d+', filename)
        return int(imageID[0])
    except:
        raise NotImplementedError('Filename %s is supposed to contain an integer.'%(filename))


def convert(xml_list, xml_dir, json_file, k_shot):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    supervise_log = {}
    for line in list_fp:
        # 每类只随机取k_shot个,随机，随机
        line, label = line.strip().split(',')
        xml_name = line.replace(line[:line.find("/")+1], "").replace("jpg", "xml")
        xml_f = os.path.join(xml_dir, xml_name)
        if label not in supervise_log:
            supervise_log[label] = 0
        if supervise_log[label] >= k_shot and k_shot != -1:
            continue
        supervise_log[label] += 1

        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = xml_name.replace("xml", "jpg")
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':bnd_id}
        json_dict['images'].append(image)
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   bnd_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Convert pascol voc annotation xml to COCO json format.')
    )
    parser.add_argument(
        '--XML_LIST', 
        default="./metadata/C45V2/train/class_labels.txt",
        type=str,
        help='The list of xml file names to convert.'
    )
    parser.add_argument(
        '--XML_DIR', 
        default="./dataset/C45V2/Labels",
        type=str,
        help='The place where all xmls located.'
    )
    parser.add_argument(
        '--OUTPUT_JSON', 
        default="./metadata/C45V2/train/C45v2_coco_5_train.json",
        type=str,
        help='The output json file.'
    )
    parser.add_argument(
        '--K_SHOT', 
        default=5,
        type=int,
        help='The number of images to convert. -1 means all.'
    )
    args = parser.parse_args()

    convert(args.XML_LIST, args.XML_DIR, args.OUTPUT_JSON, args.K_SHOT)
