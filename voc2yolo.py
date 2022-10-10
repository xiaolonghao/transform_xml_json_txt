import os
import shutil
from glob import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    # TODO 为啥-1
    # x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    x, y, w, h = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_voc_to_yolo(xml_paths, img_root, save_label_root, save_image_root):
    for xml_path in tqdm(xml_paths, total=len(xml_paths)):
        name = os.path.basename(xml_path).split('.')[0]
        img_path = os.path.join(img_root, name + '.jpg')
        save_label_path = os.path.join(save_label_root, name + '.txt')
        save_img_path = os.path.join(save_image_root, name + '.jpg')

        with open(save_label_path, 'w') as out_file:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls in classes and int(obj.find('difficult').text) != 1:
                    xmlbox = obj.find('bndbox')
                    bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                    cls_id = classes.index(cls)  # class id
                    out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
        shutil.copy(img_path, save_img_path)


if __name__ == "__main__":
    xml_root = 'demo_datas/xml/Annotations'
    img_root = 'demo_datas/xml/JPEGImages'
    xml_paths = glob(os.path.join(xml_root, '*.xml'))
    img_paths = glob(os.path.join(img_root, '*.jpg'))

    save_label_root = 'temp/labels'
    save_image_root = 'temp/images'
    os.makedirs(save_label_root, exist_ok=True)
    os.makedirs(save_image_root, exist_ok=True)

    convert_voc_to_yolo(xml_paths, img_root, save_label_root, save_image_root)
