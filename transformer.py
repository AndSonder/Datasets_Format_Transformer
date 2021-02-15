from __future__ import absolute_import

import os
import xml.etree.ElementTree as ET
import json
import settings
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os


class Transformer:
    def __init__(self):
        pass

    def xywh2xyxy(self, box):
        """
        turn xywh bow to xyxy box
        :param box: (x,y,w,h)
        :return: (x_min,y_min,x_max,y_max) , type: int
        """
        x_min = box[0] - box[2] / 2
        x_max = box[0] + box[2] / 2
        y_min = box[1] - box[3] / 2
        y_max = box[1] + box[3] / 2
        return map(int, (x_min, y_min, x_max, y_max))

    def xywh2float(self, shape, box):
        """
        turn xywh to floating xywh
        :param shape: [height,weight]
        :param box: (x,y,w,h)
        :return: (x,y,w,h)
        """
        dw = 1. / shape[1]
        dh = 1. / shape[0]
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def xyxy2xywh(self, box):
        """
        turn xyxy to xywh
        :param box: (x_min,y_min,x_max,y_max)
        :return: (x,y,w,h) type: int
        """
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        return map(int, (x, y, w, h))

    def __get(self, root, name):
        vars = root.findall(name)
        return vars

    def __get_and_check(self, root, name, length):
        vars = root.findall(name)
        if len(vars) == 0:
            raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
        if length == 1:
            vars = vars[0]
        return vars

    @staticmethod
    def __get_filename_as_int(filename):
        try:
            filename = os.path.splitext(filename)[0]
            return int(filename)
        except:
            raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))

    def voc2json(self, xml_dir, json_path):
        """
        voc format to json format
        :param xml_dir: the dir path of xml files
        :param json_path: the path of the output json file to save
        """
        xml_list = os.listdir(xml_dir)
        json_dict = {"images": [], "type": "instances", "annotations": [],
                     "categories": []}
        categories = settings.class2id
        bnd_id = settings.START_BOUNDING_BOX_ID
        for line in xml_list:
            print("Processing %s" % (line))
            xml_f = os.path.join(xml_dir, line)
            tree = ET.parse(xml_f)
            root = tree.getroot()
            path = self.__get(root, 'path')
            if len(path) == 1:
                filename = os.path.basename(path[0].text)
            elif len(path) == 0:
                filename = self.__get_and_check(root, 'filename', 1).text
            else:
                raise NotImplementedError('%d paths found in %s' % (len(path), line))
            # The filename must be a number
            image_id = self.__get_filename_as_int(filename)
            size = self.__get_and_check(root, 'size', 1)
            width = int(self.__get_and_check(size, 'width', 1).text)
            height = int(self.__get_and_check(size, 'height', 1).text)
            image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
            json_dict['images'].append(image)
            # Cruuently we do not support segmentation
            for obj in self.__get(root, 'object'):
                category = self.__get_and_check(obj, 'name', 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = self.__get_and_check(obj, 'bndbox', 1)
                xmax, xmin, ymax, ymin = self.__get_box_info(bndbox)
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
        json_fp = open(json_path, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()

    def __get_box_info(self, bndbox):
        xmin = int(self.__get_and_check(bndbox, 'xmin', 1).text) - 1
        ymin = int(self.__get_and_check(bndbox, 'ymin', 1).text) - 1
        xmax = int(self.__get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(self.__get_and_check(bndbox, 'ymax', 1).text)
        return xmax, xmin, ymax, ymin

    def json2yolo(self, yolo_root, json_root, data_type, is_flir=False):
        os.makedirs(os.path.join(yolo_root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_root, 'labels'), exist_ok=True)
        images_dir = os.path.join(yolo_root, 'images')
        labels_dir = os.path.join(yolo_root, 'labels')
        ann_file = '%s/annotations/instances_%s.json' % (json_root, data_type)
        classes = settings.class2id
        coco = COCO(ann_file)  # load annotation file
        list_file = open('%s/%s.txt' % (yolo_root, data_type), 'w')  # yolo txt file

        img_ids = coco.getImgIds()  # get images' id
        cat_ids = coco.getCatIds()  # get images' classes id

        for imgId in tqdm(img_ids):
            obj_count = 0
            Img = coco.loadImgs(imgId)[0]  # load image info
            filename = Img['file_name']  # name
            if is_flir:
                filename = self.__get_image_name(filename)
            width = Img['width']  # width
            height = Img['height']  # height
            ann_ids = coco.getAnnIds(imgIds=imgId, catIds=cat_ids, iscrowd=None)
            for annId in ann_ids:
                anns = coco.loadAnns(annId)[0]  # load annotation
                cat_id = anns['category_id']
                cat = coco.loadCats(cat_id)[0]['name']

                if cat in classes:
                    obj_count = obj_count + 1
                    if is_flir:
                        out_file = open(os.path.join(labels_dir, filename[:-5] + '.txt'), 'a')
                    else:
                        out_file = open(os.path.join(labels_dir, filename[:-4] + '.txt'), 'a')
                    cls_id = classes[cat]  # get the classes' id
                    box = anns['bbox']
                    size = [height, width]
                    bb = self.xywh2float(size, box)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                    out_file.close()

            if obj_count > 0:
                src_img = os.path.join(json_root, '%s/%s' % (data_type, filename))
                dst_img = os.path.join(images_dir, filename)
                try:
                    shutil.copy(src_img, dst_img)
                    list_file.write(os.path.join(yolo_root, 'images/%s\n' % filename))
                except:
                    print(src_img)

        list_file.close()

    def Flir2COCO(self, filr_root, coco_root):
        """
        Filr to coco
        :param filr_root: the root path of filr dataset
        :param coco_root: the root path you want to produce
        :return:
        """
        # create coco dirs
        os.makedirs(os.path.join(coco_root, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(coco_root, 'train_thermal'), exist_ok=True)
        os.makedirs(os.path.join(coco_root, 'train_rgb'), exist_ok=True)
        os.makedirs(os.path.join(coco_root, 'val_thermal'), exist_ok=True)
        os.makedirs(os.path.join(coco_root, 'val_rgb'), exist_ok=True)
        # cp json file
        os.system("sudo cp %s %s" % (os.path.join(filr_root, 'train', 'thermal_annotations.json'),
                                     os.path.join(coco_root, 'annotations', 'instances_train_thermal.json')))
        os.system("sudo cp %s %s" % (os.path.join(filr_root, 'train', 'thermal_annotations.json'),
                                     os.path.join(coco_root, 'annotations', 'instances_train_rgb.json')))
        os.system("sudo cp %s %s" % (os.path.join(filr_root, 'val', 'thermal_annotations.json'),
                                     os.path.join(coco_root, 'annotations', 'instances_val_thermal.json')))
        os.system("sudo cp %s %s" % (os.path.join(filr_root, 'val', 'thermal_annotations.json'),
                                     os.path.join(coco_root, 'annotations', 'instances_val_rgb.json')))
        self.__copy_images(filr_root, coco_root, ['train', 'RGB', 'train_rgb'])
        self.__copy_images(filr_root, coco_root, ['train', 'thermal_8_bit', 'train_thermal'])
        self.__copy_images(filr_root, coco_root, ['val', 'RGB', 'val_rgb'])
        self.__copy_images(filr_root, coco_root, ['val', 'thermal_8_bit', 'val_thermal'])

    def __copy_images(self, filr_root, coco_path, mode):
        images = os.listdir(os.path.join(filr_root, mode[0], mode[1]))
        for item in tqdm(images):
            image_name = self.__get_image_name(item)
            os.system("sudo cp %s %s" % (os.path.join(filr_root, mode[0], mode[1], item),
                                         os.path.join(coco_path, mode[2], image_name)))

    def __get_image_name(self, name):
        return name.split('_')[-1]

    def __get_txt_path(self, image_path):
        txt_path = os.path.join("/".join(image_path.split('/')[:-2]), 'labels')
        txt_path = os.path.join(txt_path, image_path.split('/')[-1].split('.')[0] + '.txt')
        return txt_path

    def __statistics_class_number(self, txt_path):
        with open(txt_path, 'r') as f:
            classes = f.readlines()
        result = dict()
        for key in settings.class2id:
            result[key] = 0
        for class_ in classes:
            result[settings.id2class[int(class_[0])]] += 1
        return result

    def select_yolo_images(self, txt_path, output_path):
        """
        select images
        """
        with open(txt_path, 'r') as f:
            images = f.readlines()
        useful_images = ""
        for image in tqdm(images):
            txt = self.__get_txt_path(image)
            classes_number = self.__statistics_class_number(txt)
            if 4 < classes_number['person'] < 10:
                useful_images += image
        with open(output_path, 'w') as f:
            f.write(useful_images)


if __name__ == '__main__':
    t = Transformer()
    # t.Flir2COCO('/home/corona/datasets/Flir/FLIR_ADAS_1_3/', '/home/corona/datasets/Flir_COCO')
    # coco = COCO('/home/corona/datasets/coco2017/annotations/instances_val2017.json')  # load annotation file
    #
    # img_ids = coco.getImgIds()
    # cat_ids = coco.getCatIds()
    # print(coco.loadImgs(img_ids[0])[0])
    # t.json2yolo('/home/corona/datasets/flir_yolo/val', '/home/corona/datasets/Flir_COCO', 'val_thermal',
    #             is_flir=True)
    t.select_yolo_images('/home/corona/datasets/flir_yolo/val/val_thermal.txt',
                         '/home/corona/datasets/flir_yolo/val/val.txt')
