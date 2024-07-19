import cv2
import numpy as np
import sys

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def find_rois_from_contours(binary):    
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

def template_match(binary_mask, id):

    mask_with_rect = binary_mask.copy()
    template = cv2.imread(f'template/{id}.PNG', cv2.IMREAD_GRAYSCALE)
    thresh, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    tx,ty,tw,th = cv2.boundingRect(template)
    template = template[ty:ty+th, tx:tx+tw]
    angles = [0, 45, 90, 135, 180, 225, 270, 315] 
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]  
    rotated_templates = [rotate_image(template, angle) for angle in angles]
    scaled_rotated_templates = []
    for rotated_template in rotated_templates:
        for scale in scales:
            scaled_rotated_templates.append(cv2.resize(rotated_template, None, fx=scale, fy=scale))


    rois = find_rois_from_contours(binary_mask)
    
    for roi in rois:
        best_match_value = float('-inf')
        best_match_location = None
        best_match_size = None
        x, y, w, h = roi
        roi_src = binary_mask[y:y+h, x:x+w]

        for scaled_rotated_template in scaled_rotated_templates:
            if scaled_rotated_template.shape[0] <= roi_src.shape[0] and scaled_rotated_template.shape[1] <= roi_src.shape[1]:
                res = cv2.matchTemplate(roi_src, scaled_rotated_template, cv2.TM_CCORR_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > best_match_value:
                    best_match_value = max_val
                    best_match_location = max_loc
                    best_match_size = (w, h)
                    besttemp = scaled_rotated_template

                if best_match_value is not None:  # 设置一个阈值来确定匹配是否成功
                    top_left = best_match_location
                    w, h = best_match_size
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    # 调整 best_temp 的尺寸以匹配目标区域
                    resized_best_temp = cv2.resize(besttemp, (w, h))  
                    mask_with_rect[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = resized_best_temp
    
        return mask_with_rect
