import cv2

__all__ = ['SimplePatternDetector']


# define valid contour parameter limits in pixels
# MANUALLY EDIT
MIN_AREA = 10  # 10 x 10
MAX_AREA = 10000  # 100 x 100

# define aspect ratio width/height
# MANUALLY EDIT
MIN_ASPECT_RATIO = 0.25  # 1/5
MAX_ASPECT_RATIO = 5.0

# define thresholds for Otsu method
# MANUALLY EDIT
OTSU_SENSITIVITY = 22
OTSU_LOW_THRESH = 45
OTSU_HIGH_THRESH = 255


class SimplePatternDetector:
    def _truncate(self, val, decimals=0):
        multiplier = 10 ** decimals
        return int(1.0 * val * multiplier) / int(multiplier)

    def _preview_image(self, dsc, im, timeout=2000, debug=False):
        if debug:
            cv2.namedWindow(dsc)
            cv2.imshow(dsc, im)
            cv2.waitKey(timeout)
            cv2.destroyAllWindows()

    def _calc_difference_otsu(self, im, bg, debug=False):
        self._preview_image('original', im, debug=debug)

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self._preview_image('image gray', im_gray, debug=debug)

        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        self._preview_image('background gray', bg_gray, debug=debug)

        # compute difference
        diff_gray = cv2.absdiff(bg_gray, im_gray)
        self._preview_image('pre-diff', diff_gray, debug=debug)

        diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        self._preview_image('pre-diff blured', diff_gray_blur, debug=debug)

        # find Otsu's threshold image
        ret, otsu_thresh = cv2.threshold(diff_gray_blur, OTSU_LOW_THRESH, OTSU_HIGH_THRESH,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._preview_image('otsu threshold', otsu_thresh, debug=debug)

        if ret < OTSU_SENSITIVITY:
            # discard image
            # make the difference zero by subtracting backgrounds
            diff = cv2.absdiff(bg_gray, bg_gray)
        else:
            diff = cv2.GaussianBlur(otsu_thresh, (5, 5), 0)
        self._preview_image("image threshold", diff, debug=debug)

        return diff

    def _identify_valid_contours(self, contours, width, height):
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
            if MIN_AREA < contour_area < MAX_AREA:
                # discard as noise on aspect ratio
                if MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                    # discard if at the edge
                    if not edge_noise:
                        valid_ids.append(i)
        return valid_ids

    def _detect_patterns(self, im, bg, external_contours=True, debug=False):
        # compute image difference
        diff = self._calc_difference_otsu(im, bg, debug=debug)

        # find the contours
        # use RETR_EXTERNAL for only outer contours
        # use RETR_TREE for all the hierarchy
        if external_contours:
            contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # identify the valid contours
        contour_ids = self._identify_valid_contours(contours, im.shape[1], im.shape[0])
        pattern_count = len(contour_ids)

        return pattern_count, contours, contour_ids

    def _output_patterns(self, im, pattern_count, contours, valid_ids, debug=False):
        im_out = im.copy()
        points_detected = []

        if len(valid_ids) > 0:
            for i, idx in enumerate(valid_ids):
                contour = contours[idx]

                # get rectangle bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # get centroid
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # draw bounding box
                im_out = cv2.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # draw center of pattern
                im_out = cv2.circle(im_out, (cx, cy), 2, (0, 255, 0), 2)
                # draw the text
                im_out = cv2.putText(im_out, f'Pt {i}', (x - w, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(im_out, f'{self._truncate(cx, 2)},{self._truncate(cy, 2)}', (x - w, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                points_detected.append([x, y, w, h, cx, cy])
        self._preview_image('pattern detected', im_out, debug=debug)
        return pattern_count, points_detected, im_out

    def run_detection(self, im, bg, debug=False):
        pattern_count, contours, contour_ids = self._detect_patterns(im, bg, debug=debug)
        pattern_count, points_detected, im_out = self._output_patterns(im, pattern_count, contours, contour_ids, debug=debug)
        return pattern_count, points_detected, im_out

    def test_detection(self, im_path, bg_path, debug=True):
        im = cv2.imread(im_path)
        bg = cv2.imread(bg_path)
        self.run_detection(im, bg, debug)


if __name__ == '__main__':
    pr = SimplePatternDetector()
    pr.test_detection('./fg.jpg', './bg.png')
