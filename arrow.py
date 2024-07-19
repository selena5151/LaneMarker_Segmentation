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
    def __init__(self, template_path, start_angle=0, end_angle=360, first_step=30, threshold=0.7):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        _, self.template = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.first_step = first_step
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

    def rotate_match(self, src):    
        match_results = []        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.match_template_at_angle, src, angle) for angle in np.arange(self.start_angle, self.end_angle + self.first_step, self.first_step)]
            for future in futures:
                result = future.result()
                if result:
                    match_results.append(result)
        
        return match_results

    def match_template_at_angle(self, src, angle):
    
        rotated_img = self.image_rotate(self.template, angle)
        result = cv2.matchTemplate(src, rotated_img, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(result)
        
        if maxval > self.threshold:
            final_point = (int(maxloc[0]), int(maxloc[1]))
            points = self.get_rotate_points((rotated_img.shape[1], rotated_img.shape[0]), angle)
            for i in range(len(points)):
                points[i][0] += final_point[0]
                points[i][1] += final_point[1]
            return MatchResult(points, angle, maxval, final_point, (rotated_img.shape[1], rotated_img.shape[0]))
        return None

    def draw_and_replace_arrows(self, img, match_results):
        for match_result in match_results:
            # 覆蓋偵測區域
            x, y = match_result.location
            w, h = match_result.size
            rotated_template = self.image_rotate(self.template, match_result.angle)
            resized_template = cv2.resize(rotated_template, (w, h))
            img[y:y+h, x:x+w] = resized_template

        return img