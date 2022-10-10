import os
import cv2
import shutil
from glob import glob
from tqdm import tqdm
from lxml import etree, objectify

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def save_anno_to_xml(filename, size, objs, save_path):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        # E.folder("DATA"),
        E.filename(filename),
        E.source(
            E.database("The VOC Database"),
            E.annotation("PASCAL VOC"),
            E.image("flickr")
        ),
        E.size(
            E.width(size[1]),
            E.height(size[0]),
            E.depth(size[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose("Unspecified"),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[1][0]),
                E.ymin(obj[1][1]),
                E.xmax(obj[1][2]),
                E.ymax(obj[1][3])
            )
        )
        anno_tree.append(anno_tree2)
    anno_path = os.path.join(save_path, filename.split('.')[0] + ".xml")
    etree.ElementTree(anno_tree).write(anno_path, pretty_print=True)


def cxcywhn2xyxy(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    xmax = (bbox[0] + bbox[2] / 2.) * size[1]
    ymax = (bbox[1] + bbox[3] / 2.) * size[0]
    box = [xmin, ymin, xmax, ymax]
    return list(map(int, box))


def convert_yolo_to_voc(txt_paths, img_root, save_xml_root, save_image_root):
    category_id = dict((k, v) for k, v in enumerate(classes))

    for txt_path in tqdm(txt_paths, total=len(txt_paths)):
        img_name = os.path.basename(txt_path).split('.')[0] + '.jpg'
        img_path = os.path.join(img_root, img_name)
        img_shape = cv2.imread(img_path).shape
        objects = []
        with open(txt_path, 'r') as f:
            for i in f.readlines():
                i = i.strip().split()
                category = int(i[0])
                category_name = category_id[category]
                bbox = cxcywhn2xyxy((i[1], i[2], i[3], i[4]), img_shape)
                obj = [category_name, bbox]
                objects.append(obj)
        save_anno_to_xml(img_name, img_shape, objects, save_xml_root)
        shutil.copyfile(img_path, os.path.join(save_image_root, img_name))


if __name__ == "__main__":
    txt_root = 'temp/labels'
    img_root = 'temp/images'
    img_paths = glob(os.path.join(img_root, '*.jpg'))
    txt_paths = glob(os.path.join(txt_root, '*.txt'))

    save_xml_root = 'temp/Annotations'
    save_image_root = 'temp/JPEGImages'
    os.makedirs(save_xml_root, exist_ok=True)
    os.makedirs(save_image_root, exist_ok=True)

    convert_yolo_to_voc(txt_paths, img_root, save_xml_root, save_image_root)
