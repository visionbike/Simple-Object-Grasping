from random import random
from pathlib import Path
import numpy as np
import cv2
import argparse


def boxes_from_YOLO(img_path: str, lbl_path: str):
    """

    :param img_path: the input image path.
    :param lbl_path: the input label path.
    :return: (img, boxes)
    """

    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    with open(lbl_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    boxes = []
    if lines != ['']:
        for line in lines:
            components = line.split(" ")
            category = components[0]
            x = int(float(components[1]) * img_width - float(components[3]) * img_width / 2)
            y = int(float(components[2]) * img_height - float(components[4]) * img_height / 2)
            h = int(float(components[4]) * img_height)
            w = int(float(components[3]) * img_width)
            boxes.append([category, x, y, w, h])
    return img, boxes


def augment_flip_verticle(img: np.array, boxes: list):
    """
    Augment image by verticle flipping

    :param img: the input image.
    :param boxes: the input bounding boxes.
    :return: (img_new, boxes_new)
    """

    boxes_new = []
    img_new = cv2.flip(img, 1)
    img_new_height, img_new_width, _ = img_new.shape
    for box in boxes:
        class_name = int(box[0])
        x, y, w, h = box[1:]
        x2 = ((img_new_width - x - w) + w / 2) / img_new_width
        h2 = h / img_new_height
        w2 = w / img_new_width
        y2 = (y + (h / 2)) / img_new_height
        boxes_new.append((class_name, round(x2, 4), round(y2, 4), round(w2, 4), round(h2, 4)))
    return img_new, boxes_new


def augment_flip_horizontal(img: np.array, boxes: list):
    """
    Augment image by horizontal flipping

    :param img: the input image.
    :param boxes: the input bounding boxes.
    :return: (img_new, boxes_new)
    """

    boxes_new = []
    img_new = cv2.flip(img, 0)
    img_new_height, img_new_width, _ = img_new.shape
    for box in boxes:
        class_name = int(box[0])
        x, y, w, h = box[1:]
        x2 = (x + (w / 2)) / img_new_width
        h2 = h / img_new_height
        w2 = w / img_new_width
        y2 = ((img_new_height - y - h) + h / 2) / img_new_height
        boxes_new.append((class_name, round(x2, 4), round(y2, 4), round(w2, 4), round(h2, 4)))
    return img_new, boxes_new


def augment_flip_horizontal_verticle(img: np.array, boxes: list):
    """
    Augment image by horizontal and verticle flipping

    :param img: the input image.
    :param boxes: the input bounding boxes.
    :return: (img_new, boxes_new)
    """

    boxes_new = []
    img_new = cv2.flip(img, -1)
    img_new_height, img_new_width, _ = img_new.shape
    for box in boxes:
        class_name = int(box[0])
        x, y, w, h = box[1:]
        x2 = ((img_new_width - x - w) + w / 2) / img_new_width
        h2 = h / img_new_height
        w2 = w / img_new_width
        y2 = ((img_new_height - y - h) + h / 2) / img_new_height
        boxes_new.append((class_name, round(x2, 4), round(y2, 4), round(w2, 4), round(h2, 4)))
    return img_new, boxes_new


def augment_brightness(img: np.array, boxes: list):
    """
    Augment image by randomly changing brightness

    :param img: the input image.
    :param boxes: the input bounding boxes.
    :return: (img_new, boxes_new)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    factor = random()
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())  # scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # reset out of range values
    img_new = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    img_height, img_width, _ = img.shape
    boxes_new = []
    for box in boxes:
        class_name = int(box[0])
        x, y, w, h = box[1:]
        x2 = (x + (w / 2)) / img_width
        h2 = h / img_height
        w2 = w / img_width
        y2 = (y + (h / 2)) / img_height
        boxes_new.append((class_name, round(x2, 4), round(y2, 4), round(w2, 4), round(h2, 4)))
    return img_new, boxes_new


def augment_data(aug_type: str, img: np.array, boxes: list):
    """
    Data augmentation with image and its bounding boxes

    :param aug_type: the augmentation type.
    :param img: the input image.
    :param boxes: the input bounding boxes.
    :return: (img_new, boxes_new)
    """

    if aug_type == 'hflip':
        return augment_flip_horizontal(img, boxes)
    elif aug_type == 'vflip':
        return augment_flip_verticle(img, boxes)
    elif aug_type == 'hvflip':
        return augment_flip_horizontal_verticle(img, boxes)
    elif aug_type == 'bright':
        return augment_brightness(img, boxes)
    else:
        raise ValueError(f'type={aug_type} is not supported!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Augmentation')
    parser.add_argument('-i', type=str, help='input train.txt')
    parser.add_argument('-t', type=str, help='input augmentation type')
    args = parser.parse_args()

    aug_types = []
    if args.t == 'all':
        aug_types = ['hflip', 'vflip', 'hvflip', 'bright']
    else:
        aug_types = [args.t]

    train_path = Path(args.i)
    if not train_path.exists():
        raise FileExistsError(f"Not found {str(train_path)}")

    with open(str(train_path), 'r') as f:
        filenames = [line.rstrip('\n') for line in f.readlines()]

    filenames_aug = []
    for fn in filenames:
        img, boxes = boxes_from_YOLO(img_path=fn, lbl_path=fn.replace('jpg', 'txt'))

        for aug_type in aug_types:
            img, boxes = augment_data(aug_type, img, boxes)
            cv2.imwrite(f'{fn[:-4]}_{aug_type}.jpg', img)
            with open(f'{fn[:-4]}_{aug_type}.txt', 'w') as f:
                if len(boxes) > 0:
                    for box in boxes:
                        f.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')
                else:
                    f.write('')
            filenames_aug.append(f'{fn[:-4]}_{aug_type}.jpg')

    filenames += filenames_aug
    filenames = [(str(fn) + '\n') for fn in filenames]
    with open(str(train_path), 'w') as f:
        f.writelines(filenames)
