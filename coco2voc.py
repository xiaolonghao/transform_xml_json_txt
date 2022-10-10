import os
import json
import shutil
from glob import glob
from tqdm import tqdm
from xml.dom import minidom


# classes = ['_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
#            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def convert_boxshape(box):
    x = box[0]  # 标注区域X坐标
    y = box[1]  # 标注区域Y坐标
    w = box[2]  # 标注区域宽度
    h = box[3]  # 标注区域高度

    xmin = int(x)
    ymin = int(y)
    xmax = int(x + w)
    ymax = int(y + h)

    return (xmin, ymin, xmax, ymax)


def get_xml(img_name, size, roi, outpath):
    '''
    传入每张图片的名称 大小 和标记点列表生成和图片名相同的xml标记文件
    :param img_name: 图片名称
    :param size:[width,height,depth]
    :param roi:[["label1",xmin,ymin,xmax,ymax]["label2",xmin,ymin,xmax,ymax].....]
    :return:
    '''
    impl = minidom.getDOMImplementation()
    doc = impl.createDocument(None, None, None)
    # 创建根节点
    orderlist = doc.createElement("annotation")
    doc.appendChild(orderlist)
    # c创建二级节点
    filename = doc.createElement("filename")
    filename.appendChild(doc.createTextNode(img_name))
    orderlist.appendChild(filename)
    # size节点
    sizes = doc.createElement("size")
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(size[0])))
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(size[1])))
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(size[2])))
    sizes.appendChild(width)
    sizes.appendChild(height)
    sizes.appendChild(depth)
    orderlist.appendChild(sizes)
    # object 节点
    for ri in roi:
        object = doc.createElement("object")
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(ri[0]))
        object.appendChild(name)
        bndbox = doc.createElement("bndbox")

        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(ri[1])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(ri[2])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(ri[3])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(ri[4])))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        orderlist.appendChild(object)
    with open(os.path.join(outpath, img_name.split('.')[0] + '.xml'), 'w') as f:
        doc.writexml(f, addindent='  ', newl='\n')


def convert_coco_to_voc(annotation_path, JPEGImage_path, save_xml_root, save_image_root):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    imgs = data['images']
    annotations = data['annotations']
    categorie = data["categories"]

    # TODO 注意json中存的categorie 标准的coco数据是id=1开始到id=90，会有supercategory 和 category
    classes = {ca['name']: ca['id'] for ca in categorie}
    classes = list(classes.keys())

    # 遍历图片json列表
    for img in tqdm(imgs, total=len(imgs)):
        filename = img['file_name']
        img_w = img['width']
        img_h = img['height']
        img_id = img['id']
        roi_list = list()
        for ann in annotations:
            if ann['image_id'] == img_id:
                box = convert_boxshape(ann['bbox'])
                # TODO 有问题
                # roi_list.append([classes[ann['category_id']], box[0], box[1], box[2], box[3]])
                roi_list.append([classes[ann['category_id'] - 1], box[0], box[1], box[2], box[3]])
        get_xml(filename, [img_w, img_h, 3], roi_list, save_xml_root)
        shutil.copyfile(os.path.join(JPEGImage_path, filename), os.path.join(save_image_root, filename))


if __name__ == "__main__":
    # annotation_path = 'demo_datas/json/annotations/instances_val2017.json'
    annotation_path = 'temp/xml2json.json'
    # img_root = 'demo_datas/json/val2017'
    img_root = 'temp/coco_image'
    img_paths = glob(os.path.join(img_root, '*.jpg'))

    save_xml_root = 'temp/Annotations'
    save_image_root = 'temp/JPEGImages'
    os.makedirs(save_xml_root, exist_ok=True)
    os.makedirs(save_image_root, exist_ok=True)

    # 传入coco的annotations.json地址，coco image文件夹 和输出的voc xml地址 image地址
    convert_coco_to_voc(annotation_path, img_root, save_xml_root, save_image_root)
