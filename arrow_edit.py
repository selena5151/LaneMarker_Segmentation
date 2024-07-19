import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class MatchResult:
    def __init__(self, points, angle, score, location, size):
        self.points = points
        self.angle = angle
        self.score = score
        self.location = location
        self.size = size

class ArrowDetector:
    def __init__(self, template_path, start_angle=0, end_angle=360, first_step=30, second_step=1, threshold=0.3):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = cv2.boundingRect(contours[0])
        self.template = template[y:y+h, x:x+w]
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.first_step = first_step
        self.second_step = second_step
        self.threshold = threshold

    @staticmethod
    def image_rotate(image, angle):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        return rotated_image

    @staticmethod
    def get_rotate_points(size, angle):
        w, h = size
        points = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        points = np.int0(cv2.transform(np.array([points]), rot_matrix))[0]
        return points.tolist()

    def rotate_match(self, src, rois=None):
        match_results = []        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.match_template_at_angle, src, angle, rois) for angle in np.arange(self.start_angle, self.end_angle + self.first_step, self.first_step)]
            for future in futures:
                result = future.result()
                if result:
                    match_results.append(result)
        
        return match_results

    def find_rois_from_contours(self, src):
        # Convert image to binary
        _, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        for contour in contours:
            # Get the minimum area rectangle for each contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            x, y, w, h = cv2.boundingRect(box)
            rois.append((x, y, w, h))
        
        return rois

    def match_template_at_angle(self, src, angle, rois=None):
        rotated_img = self.image_rotate(self.template, angle)

        match_result = None
        maxval = 0
        maxloc = None

        if rois is None:
            rois = [(0, 0, src.shape[1], src.shape[0])]

        for roi in rois:
            x, y, w, h = roi
            roi_src = src[y:y+h, x:x+w]
            rotated_img = cv2.resize(rotated_img, (w,h))
            result = cv2.matchTemplate(roi_src, rotated_img, cv2.TM_CCOEFF_NORMED)
            _, temp_maxval, _, temp_maxloc = cv2.minMaxLoc(result)

            if temp_maxval > maxval and temp_maxval > self.threshold:
                maxval = temp_maxval
                maxloc = (x, y)
                match_result = MatchResult(self.get_rotate_points((rotated_img.shape[1], rotated_img.shape[0]), angle), angle, maxval, maxloc, (rotated_img.shape[1], rotated_img.shape[0]))

        return match_result


    def draw_and_replace_arrows(self, img, match_results):
        for match_result in match_results:
            # 覆蓋偵測區域
            x, y = match_result.location
            w, h = match_result.size
            rotated_template = self.image_rotate(self.template, match_result.angle)
            resized_template = cv2.resize(rotated_template, (w, h))
            img[y:y+h, x:x+w] = resized_template
            print("success")

        return img