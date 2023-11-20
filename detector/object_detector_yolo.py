from pathlib import Path
import numpy as np
import cv2

__all__ = ['YoloObjectDetector']


class YoloObjectDetector:
    """
    Object Detector that using YOLO
    """

    def __init__(self, model_path: str, use_cuda: bool = False):
        """

        :param model_path: the path that stores YOLO model.
        :param use_cuda: whether to use CUDA. Default: False.
        """
        self.model_path = Path(model_path)
        self._load_model(use_cuda)
        self._load_class_names()
        self.IMG_WIDTH, self.IMG_HEIGHT = 0, 0

    def _load_model(self, use_cuda: bool = False):
        """
        Load the YOLO model.

        :param use_cuda: whether to use CUDA. Default: False.
        """
        self.net = cv2.dnn.readNet(str(self.model_path / 'best.onnx'))
        if use_cuda:
            print('Attempt to use CUDA')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print('Running on CPU')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _load_class_names(self):
        """
        Load the class names from `obj.names` files
        """
        with open(str(Path(self.model_path / 'obj.names')), 'r') as f:
            self.class_names = {i: name.strip() for i, name in enumerate(f.readlines())}

    def format_yolo_input(self, img: np.array):
        """
        Format the input image for YOLO image

        :param img: the input image.
        :return: the output image
        """

        self.IMG_HEIGHT, self.IMG_WIDTH, _ = img.shape
        size_max = max(self.IMG_WIDTH, self.IMG_HEIGHT)
        out = np.zeros((size_max, size_max, 3), np.uint8)
        out[0: self.IMG_HEIGHT, 0: self.IMG_WIDTH] = img
        return out

    def detect_objects(self, img: np.array, min_box_conf_thresh: float = 0.25, max_box_conf_thresh: float = 0.5, class_conf_thresh: float = 0.5):
        """
        Detect objects from the image.

        :param img: the input image.
        :param min_box_conf_thresh: the minimum box confidence threshold. Default: 0.25.
        :param max_box_conf_thresh: the maximum box confidence threshold. Default: 0.5.
        :param class_conf_thresh: the class confidence threshold. Default: 0.5.
        :return: (class_ids, box_confs, boxes)
        """

        # format the input image into YOLO format
        img = self.format_yolo_input(img)

        ratio_x = img.shape[0] / self.IMG_WIDTH
        ratio_y = img.shape[1] / self.IMG_HEIGHT

        # get predictions from YOLO
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, img.shape[:-1], swapRB=True, crop=False)
        self.net.setInput(blob)
        # the outputs are a box of vectors, which contains:
        # the bounding box (cx, cy, width, height)
        # the box confidence
        # class confidence values
        outputs = self.net.forward()

        # post-processing
        class_ids = []
        box_confs = []
        boxes = []
        # go through each detected object
        for output in outputs:
            for detection in output:
                box_conf = detection[4:]        # get the box confidence value
                # process the detected bounding box having confidence value larger than threshold,
                # otherwise, discard it
                if box_conf > min_box_conf_thresh:
                    scores = detection[5:]          # get the detection class confidence values
                    class_id = np.argmax(scores)    # get the class having the best confidence value
                    class_conf = scores[class_id]   # get the best confidence value
                    # store the bounding box for the detected objects which have confidence values larger than the threshold
                    # discard the confidence values which are smaller the threshold value
                    if class_conf > class_conf_thresh:
                        x = int((detection[0] - 0.5 * detection[2]) * ratio_x)
                        y = int((detection[1] - 0.5 * detection[3]) * ratio_y)
                        w = int(detection[2] * ratio_x)
                        h = int(detection[3] * ratio_y)
                        box = np.array([x, y, w, h])

                        class_ids.append(class_id)
                        box_confs.append(box_conf)
                        boxes.append(box)
        # do NMS to suppress incorrect boxes
        ids = cv2.dnn.NMSBoxes(boxes, box_confs, min_box_conf_thresh, max_box_conf_thresh)
        class_ids = list(map(class_ids.__getitem__, ids))
        box_confs = list(map(box_confs.__getitem__, ids))
        boxes = list(map(boxes.__getitem__, ids))

        return class_ids, box_confs, boxes

    def get_class_name(self, class_id: int):
        """
        Get the clas name given the ID.

        :param class_id: the class id.
        :return: the corresponding class name
        """

        try:
            return self.class_names[class_id]
        except KeyError:
            return -1
