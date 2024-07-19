import cv2
import numpy as np
import torch
from arrow import ArrowDetector
# from arrow_edit import ArrowDetector
from crf import apply_crf

class ImageProcessor:
    def __init__(self):
        self.arrow = {2: 5, 10: 20, 16: 36, 17: 37, 20: 43, 22: 49, 24: 51, 25: 56}
        self.area = {8: 18, 12: 23, 15: 32, 23: 50}
        # self.line = {4: 10, 5: 11, 11: 22, 13: 25, 14: 27, 19: 42}
        self.line = {5: 11, 13: 25, 14: 27, 19: 42} #without double line
        self.curve = {26: 64}
        self.word = [1, 3, 4, 7, 8, 9, 13, 15, 16, 17, 21, 24, 26, 28, 29,
                    30, 31, 33, 34, 35, 38, 40, 41, 44, 45, 47, 48, 52,
                    53, 54, 55, 57, 58, 59, 60, 61, 62, 63]
        self.singleword = [27]


    @staticmethod
    def min_area_rect_mask(binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_with_rect = np.zeros_like(binary_mask)
        for c in contours:
            # find minimize area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(mask_with_rect, [box], -1, (255, 255, 255), cv2.FILLED)
        return mask_with_rect if contours else binary_mask

    @staticmethod 
    def area_mask(binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_with_rect = np.zeros_like(binary_mask)
        for c in contours:
            # find minimize area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Calculate the area of the min area rect and the contour
            rect_area = cv2.contourArea(box)
            contour_area = cv2.contourArea(c)
        
            # 如果minarea範圍並沒有範圍大於原始contour太多就用這個
            if rect_area <= 1.5 * contour_area:
                cv2.drawContours(mask_with_rect, [box], -1, (255, 255, 255), cv2.FILLED)
            else:
                # 如果minarea範圍大於原始contour太多就用這個
                epsilon = 0.01 * cv2.arcLength(c, True)
                approx_polygon = cv2.approxPolyDP(c, epsilon, True)
                selected_vertices = [approx_polygon[i] for i in range(len(approx_polygon))]
                cv2.fillPoly(mask_with_rect, [np.array(selected_vertices)], 255)
        return mask_with_rect if contours else binary_mask

    @staticmethod
    def dilate_and(img, mask):
        img = img.detach().cpu().numpy()
        h = mask.shape[0]
        w = mask.shape[1]
        img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, (w, h))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = (img_gray * 255).astype(np.uint8)
        thresh, binaryImage = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate:
        kernel_img = np.ones((5, 5), np.uint8)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel_img, iterations=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask_dilate = cv2.dilate(mask, kernel)
        mask_result = cv2.bitwise_and(binaryImage, mask_dilate)
        return mask_result

    @staticmethod
    def pre_crf(img, r):
        np_r = r.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        crf_result = apply_crf(img, np_r)
        return crf_result
    

    @staticmethod
    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]

    def process_fitimg(self, ori_img, r, process_type="origin"):
        device = r.device
        crf_pre = self.pre_crf(ori_img, r)
        # max_cls = self.pre_crf(ori_img, r)
        max_cls = r.argmax(dim=0).detach().cpu()

        pred = np.array(max_cls, dtype=int)

        refined_r = torch.zeros_like(r)

        for class_id in np.unique(pred):
            if class_id == 0:
                continue

            if process_type == "origin":
                # Attempt to find the corresponding key in the dictionaries
                mapped_id = self.get_key(self.area, class_id) + self.get_key(self.line, class_id) + self.get_key(self.curve, class_id) + self.get_key(self.arrow, class_id)
                # If a mapping is found, use the first one, otherwise use the original class_id
                if mapped_id:
                    mapped_id = mapped_id[0]
                else:
                    mapped_id = -1
                use_word = "origin"
            elif process_type == "noword":
                mapped_id = class_id
                use_word = "no"
            elif process_type == "singleword":
                mapped_id = class_id
                use_word = "single"

            binary_mask = (pred == class_id).astype(np.uint8) * 255

            if mapped_id in self.area:
                if mapped_id == 15: #crosswalk
                    # Opening
                    kernel = np.ones((7, 7), np.uint8) 
                    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) 
                    # minarea
                    minarea = self.area_mask(opening)
                    refined_r[class_id] = torch.tensor(minarea, device=device).float() / 255
                else:
                    # Opening
                    kernel = np.ones((3, 3), np.uint8) 
                    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) 
                    # minarea
                    minarea = self.min_area_rect_mask(opening)
                    refined_r[class_id] = torch.tensor(minarea, device=device).float() / 255
            elif mapped_id in self.line:
                # Opening
                kernel = np.ones((3, 3), np.uint8) 
                opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) 
                # minarea
                min_line = self.min_area_rect_mask(opening)
                refined_r[class_id] = torch.tensor(min_line, device=device).float() / 255
            elif mapped_id in self.arrow:
                # #arrow template
                # arrow_detector = ArrowDetector(f'template/{self.arrow[mapped_id]}.PNG')
                # arrow_area = arrow_detector.rotate_match(binary_mask)
                # arrow_area = arrow_detector.draw_and_replace_arrows(binary_mask, arrow_area)
                # refined_r[class_id] = torch.tensor(arrow_area, device=device).float() / 255

                # crf
                binary_mask = (crf_pre == class_id).astype(np.uint8) * 255
                refined_r[class_id] = torch.tensor(binary_mask, device=device).float() / 255

            elif mapped_id in self.singleword and use_word=="single":  # single word label
                # Opening
                kernel = np.ones((3, 3), np.uint8) 
                opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel) 
                # minarea
                minarea_word = self.min_area_rect_mask(opening)
                refined_r[class_id] = torch.tensor(minarea_word, device=device).float() / 255
            elif mapped_id == -1 and class_id in self.word and use_word=="origin":  # word label
                # minarea
                minarea_word = self.min_area_rect_mask(binary_mask)
                refined_r[class_id] = torch.tensor(minarea_word, device=device).float() / 255
            else:
                refined_r[class_id] = torch.tensor(binary_mask, device=device).float() / 255

        return refined_r

    def opencv_fitimg(self, ori_img, r):
        return self.process_fitimg(ori_img, r, process_type="origin")

    def noword_fitimg(self, ori_img, r):
        return self.process_fitimg(ori_img, r, process_type="noword")

    def singleword_fitimg(self, ori_img, r):
        return self.process_fitimg(ori_img, r, process_type="singleword")
