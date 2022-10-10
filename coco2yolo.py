import os
import json
import shutil
from glob import glob
from tqdm import tqdm


def parse_coco(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]
        # 标准的coco数据是id=1开始到id=90，会有supercategory 和 category
        categories = data["categories"]
    infos = {}
    for image in images:
        infos[image["id"]] = image
        infos[image["id"]]["bbox_cat"] = []
    for anno in annotations:
        infos[anno["image_id"]]["bbox_cat"].append({"box": anno["bbox"], "cat": anno["category_id"]})
    return infos


def save_labels(infos, w, h, save_path):
    with open(save_path, "w") as f:
        for info in infos:
            # 注意，json中 anno["category_id"] 从1开始 所以要-1
            box = info["box"]
            # TODO 需不需要-1
            cx, cy, dw, dh = (box[0] + box[2] / 2.0) / w, (box[1] + box[3] / 2.0) / h, box[2] / w, box[3] / h
            f.write(" ".join([str(a) for a in [info["cat"] - 1, cx, cy, dw, dh]]) + '\n')


def convert_coco_to_yolo(json_path, img_root, save_txt_root, save_image_root):
    infos = parse_coco(json_path).values()
    for info in tqdm(infos, total=len(infos)):
        img_name = info["file_name"]
        save_txt_path = os.path.join(save_txt_root, img_name.split('.')[0] + '.txt')
        save_labels(info["bbox_cat"], info["width"], info["height"], save_txt_path)
        shutil.copyfile(os.path.join(img_root, img_name), os.path.join(save_image_root, img_name))


if __name__ == '__main__':
    annotation_path = 'temp/xml2json.json'
    img_root = 'temp/coco_image'
    img_paths = glob(os.path.join(img_root, '*.jpg'))

    save_txt_root = 'temp/labels'
    save_image_root = 'temp/images'
    os.makedirs(save_txt_root, exist_ok=True)
    os.makedirs(save_image_root, exist_ok=True)

    convert_coco_to_yolo(annotation_path, img_root, save_txt_root, save_image_root)
