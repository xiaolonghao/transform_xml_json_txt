import os
import cv2
import json
import shutil
from glob import glob
from tqdm import tqdm
import os.path as osp

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def addCatItem(category_dict):
    # TODO 注意json中存的categorie 标准的coco数据是id=1开始到id=90，会有supercategory 和 category
    category_items = []
    for k, v in category_dict.items():
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = int(k)
        category_item['name'] = v
        category_items.append(category_item)
    return category_items


def addImgItem(file_name, size, image_id):
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size[1]
    image_item['height'] = size[0]
    image_item['license'] = None
    image_item['flickr_url'] = None
    image_item['coco_url'] = None
    return image_item, image_id


def addAnnoItem(image_id, category_id, bbox, annotation_id):
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    return annotation_item, annotation_id


def cxcywhn2xywh(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    box = (xmin, ymin, w, h)
    return list(map(int, box))


def convert_yolo_to_coco(txt_paths, img_root, save_json_path, save_image_root):
    image_id = 000000
    annotation_id = 0
    images = []
    annotations = []
    category_id = dict((k, v) for k, v in enumerate(classes))
    category_item = addCatItem(category_id)
    for txt_path in tqdm(txt_paths, total=len(txt_paths)):
        img_name = os.path.basename(txt_path).split('.')[0] + '.jpg'
        img_path = os.path.join(img_root, img_name)
        img_shape = cv2.imread(img_path).shape
        image_item, image_id = addImgItem(img_name, img_shape, image_id)
        images.append(image_item)
        with open(txt_path, 'r') as fid:
            for i in fid.readlines():
                i = i.strip().split()
                category = int(i[0])
                bbox = cxcywhn2xywh((i[1], i[2], i[3], i[4]), img_shape)
                annotation_item, annotation_id = addAnnoItem(image_id, category, bbox, annotation_id)
                annotations.append(annotation_item)
        shutil.copyfile(img_path, os.path.join(save_image_root, img_name))

    final_result = {"images": images, "annotations": annotations, "categories": category_item}
    os.makedirs(osp.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, 'w') as f:
        json.dump(final_result, f)


if __name__ == '__main__':
    txt_root = 'temp/labels'
    img_root = 'temp/images'
    img_paths = glob(os.path.join(img_root, '*.jpg'))
    txt_paths = glob(os.path.join(txt_root, '*.txt'))
    save_json_path = 'temp/yolo2coco.json'
    save_image_root = 'temp/coco_image'
    os.makedirs(save_image_root, exist_ok=True)
    convert_yolo_to_coco(txt_paths, img_root, save_json_path, save_image_root)
