import os
import json
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image
import os.path as osp
import xml.etree.ElementTree as ET

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# TODO 注意索引0 or 1开始
label_ids = {name: i + 1 for i, name in enumerate(classes)}


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1], points[2] + points[0], points[3] + points[1],
            points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        area = w * h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [xmin, ymin, w, h],
            "category_id": category_id,
            "id": anno_id,
            "ignore": 0})
        anno_id += 1
    return annotation, anno_id


def convert_voc_to_coco(img_path, xml_path, out_file, save_image_path=None):
    images = []
    annotations = []

    img_id = 1
    anno_id = 1
    for img_path in tqdm(img_path, total=len(img_path)):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1
        if save_image_path is not None:
            shutil.copy(img_path, osp.join(save_image_path, img_name))
    # TODO 注意json中存的categorie 标准的coco数据是id=1开始到id=90，会有supercategory 和 category
    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(final_result, f)
    return annotations


if __name__ == '__main__':
    # 原图像路径
    img_paths = glob(osp.join('demo_datas/xml/JPEGImages/*.jpg'))
    # 原xml文件夹
    xml_root = 'demo_datas/xml/Annotations'
    save_json_path = 'temp/xml2json.json'
    save_image_path = 'temp/coco_image'
    os.makedirs(save_image_path, exist_ok=True)

    convert_voc_to_coco(img_paths, xml_root, save_json_path, save_image_path)
