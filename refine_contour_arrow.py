"""
refine contour for single word and arrow dataset
"""
import cv2
import numpy as np
import torch
from arrow import ArrowDetector
from crf import apply_crf

class ArrowProcessor:
    def __init__(self):
        self.arrow = {2}
        self.area = {8, 11, 14, 18}
        self.line = {5, 13} 
        self.singleword = [20]

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
        np_r = r.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        crf_result = apply_crf(img, np_r)
        return crf_result


    def process_fitimg(self, ori_img, r):
        device = r.device
        crf_pre = self.pre_crf(ori_img, r)
        max_cls = r.argmax(dim=0).detach().cpu()
        pred = np.array(max_cls, dtype=int)
        refined_r = torch.zeros_like(r)

        for class_id in np.unique(pred):
            if class_id == 0:
                continue

            binary_mask = (pred == class_id).astype(np.uint8) * 255

            if class_id in self.area:
                if class_id == 14: #crosswalk
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
            elif class_id in self.line:
                # Opening
                kernel = np.ones((3, 3), np.uint8) 
                opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                # minarea
                min_line = self.min_area_rect_mask(opening)
                refined_r[class_id] = torch.tensor(min_line, device=device).float() / 255
            elif class_id in self.arrow:
                # crf
                binary_mask = (crf_pre == class_id).astype(np.uint8) * 255
                refined_r[class_id] = torch.tensor(binary_mask, device=device).float() / 255

            elif class_id in self.singleword:
                # Opening
                kernel = np.ones((3, 3), np.uint8) 
                opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                # minarea
                minarea_word = self.min_area_rect_mask(opening)
                refined_r[class_id] = torch.tensor(minarea_word, device=device).float() / 255
            else:
                refined_r[class_id] = torch.tensor(binary_mask, device=device).float() / 255

        return refined_r

    def arrow_fitimg(self, ori_img, r):
        return self.process_fitimg(ori_img, r)
