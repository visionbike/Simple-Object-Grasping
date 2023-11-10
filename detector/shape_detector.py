import numpy as np
import cv2

__all__ = ['ShapeDetector']


class ShapeDetector:
    """
    Shape detector using the image difference method
    """

    def _truncate(self, val, decimals=0):
        multiplier = 10 ** decimals
        return int(1.0 * val * multiplier) / int(multiplier)

    def _preview_image(self, dsc, im, timeout=2000, debug=False):
        if debug:
            cv2.namedWindow(dsc)
            cv2.imshow(dsc, im)
            cv2.waitKey(timeout)
            cv2.destroyAllWindows()

    def _calc_difference_otsu(self,
                              im: np.array, bg: np.array,
                              min_thresh: float = 45, max_thresh: float = 255,
                              sensitivity: float = 22,
                              debug: bool = False):
        """
        Calculate the image detector using Otsu threshold method.

        :param im: the input image.
        :param bg: the background image.
        :param min_thresh: Otsu low threshold value. Default: 45.
        :param max_thresh: Otsu low threshold value. Default: 255.
        :param sensitivity: threshold for the difference. Default: 22.
        :param debug: whether to visualize the result image. Default: False.
        """

        self._preview_image('Input', im, debug=debug)

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self._preview_image('Input Gray', im_gray, debug=debug)

        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        self._preview_image('Background Gray', bg_gray, debug=debug)

        # compute difference
        diff_gray = cv2.absdiff(bg_gray, im_gray)
        self._preview_image('Pre-diff', diff_gray, debug=debug)

        diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        self._preview_image('Pre-diff blured', diff_gray_blur, debug=debug)

        # find Otsu's threshold image
        ret, otsu_thresh = cv2.threshold(diff_gray_blur, min_thresh, max_thresh,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._preview_image('otsu threshold', otsu_thresh, debug=debug)

        if ret < sensitivity:
            # discard image
            # make the difference zero by subtracting backgrounds
            diff = cv2.absdiff(bg_gray, bg_gray)
        else:
            diff = cv2.GaussianBlur(otsu_thresh, (5, 5), 0)
        self._preview_image("image threshold", diff, debug=debug)

        return diff

    def _identify_valid_contours(self,
                                 contours: np.array,
                                 width: int, height: int,
                                 min_ratio: float, max_ratio: float,
                                 min_area: float, max_area: float):
        """
        Detect the valid contour that is not corner or exceeded in area.

        :param contours: the contour list.
        :param width: the image width,
        :param height: the image height.
        :param min_ratio: min aspect ratio threshold.
        :param max_ratio: max aspect ratio threshold.
        :param min_area: min area threshold.
        :param max_area: max area threshold.
        :return: valid id list.
        """
        valid_ids = []
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            # compute aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = 1.0 * w / h

            # flag as edge_noise if th pattern is at a corner
            edge_noise = False
            if x == 0:
                edge_noise = True
            if y == 0:
                edge_noise = True
            if (x + w) == width:
                edge_noise = True
            if (y + h) == height:
                edge_noise = True

            # discard noise with measure if area not within area thresholds
            if min_area < contour_area < max_area:
                # discard as noise on aspect ratio
                if min_ratio < aspect_ratio < max_ratio:
                    # discard if at the edge
                    if not edge_noise:
                        valid_ids.append(i)
        return valid_ids

    def detect_shape(self,
                     im: np.array, bg: np.array,
                     min_thresh: float = 45, max_thresh: float = 255,
                     sensitivity: float = 22,
                     min_ratio: float = 0.25, max_ratio: float = 5.0,
                     min_area: float = 200, max_area: float = 900000,
                     external_contour: bool = True,
                     debug: bool = False):
        """
        Detect valid contours with their centroids.

        :param im: the input image.
        :param bg: the background image.
        :param min_thresh: Otsu low threshold value. Default: 45.
        :param max_thresh: Otsu low threshold value. Default: 255.
        :param sensitivity: threshold for the difference. Default: 22.
        :param min_ratio: min aspect ratio threshold.
        :param max_ratio: max aspect ratio threshold.
        :param min_area: min area threshold.
        :param max_area: max area threshold.
        :param external_contour: whether to apply external contour. Default: True.
        :param debug: whether to visualize the result image. Default: False.
        :return: (contours, centroids)
        """

        # compute image difference
        diff = self._calc_difference_otsu(im, bg, min_thresh, max_thresh, sensitivity, debug)

        # find the contours
        # use RETR_EXTERNAL for only outer contours
        # use RETR_TREE for all the hierarchy
        if external_contour:
            contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # identify the valid contours
        contour_ids = self._identify_valid_contours(contours, im.shape[1], im.shape[0], min_ratio, max_ratio, min_area, max_area)

        # detect centroids
        # x, y: top left point of bounding box
        # w, h: width, height of bounding box
        # cx, cy: centroid point
        centroids = []
        for i, idx in enumerate(contour_ids):
            contour = contours[idx]

            # get rectangle bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # estimate centroid
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append([x, y, w, h, cx, cy])

        return contours, centroids
