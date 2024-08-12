"""
refine contour
origin: each word with unique label
singleword: word in same label
"""
import cv2
import numpy as np
import torch
from arrow import ArrowDetector
from crf import apply_crf

class ImageProcessor:
    def __init__(self):
        # dictionary={singleword_labelid : origin_labelid}
        self.arrow = {2: 5, 10: 20, 16: 36, 17: 37, 20: 43, 22: 49, 24: 51, 25: 56}
        self.area = {8: 18, 12: 23, 15: 32, 23: 50}
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
        
            # if minarea<=1.5 origin contour
            if rect_area <= 1.5 * contour_area:
                cv2.drawContours(mask_with_rect, [box], -1, (255, 255, 255), cv2.FILLED)
            else:
                epsilon = 0.01 * cv2.arcLength(c, True)
                approx_polygon = cv2.approxPolyDP(c, epsilon, True)
                selected_vertices = [approx_polygon[i] for i in range(len(approx_polygon))]
                cv2.fillPoly(mask_with_rect, [np.array(selected_vertices)], 255)
        return mask_with_rect if contours else binary_mask

    @staticmethod
    def pre_crf(img, r):
        #use crf to refine arrow
        np_r = r.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        crf_result = apply_crf(img, np_r)
        return crf_result
    

    @staticmethod
    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]

    def process_fitimg(self, ori_img, r, process_type="singleword"):
        device = r.device
        crf_pre = self.pre_crf(ori_img, r)
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
                # #arrow template with opencv matchtemplate
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

    def singleword_fitimg(self, ori_img, r):
        return self.process_fitimg(ori_img, r, process_type="singleword")
