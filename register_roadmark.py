"""
register road marking dataset
"""

import os
import json
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
#from mask_former import MaskFormerSemanticDatasetMapper
from collections import namedtuple

class CustomDataset:
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.json_dir = os.path.join(data_dir, "labels", split)
        self.image_dir = os.path.join(data_dir, "images", split)
        self.load_annotations()
        

    def load_annotations(self):
        self.dataset_dicts = []
        json_files = os.listdir(self.json_dir)
        for json_file in json_files:
            record = {}
            label_set = set()
            if json_file.endswith(".json"):
                image_id = os.path.splitext(json_file)[0]
                image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
                label_file = os.path.join(self.json_dir, f"{image_id}_id.png")
                with open(os.path.join(self.json_dir, json_file), "r") as f:
                    data = json.load(f)
                img_width, img_height = data["imgWidth"], data["imgHeight"]

                record["file_name"] = image_path
                record["image_id"] = image_id
                record["height"] = img_height
                record["width"] = img_width
                record["sem_seg_file_name"] = label_file

                self.dataset_dicts.append(record)

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        return self.dataset_dicts[idx]


def register_custom_dataset(name, data_dir, split):
    #65 classes
    label_name = ['unlabeled','word(place)', 'crossover crosswalk', 'word(xin)', 'word(first)', 
              'go straight', 'bicycle lane', 'word(jia)', 'word(prohibit)', 'word(song)', 
              'double yellow line', 'broken lane line', 'giveaway', 'word(tong)', 'guide sign', 
              'speed limit(50)', 'word(south)', 'speed limit(60)', 'cross hatch', 'obstacle', 
              'turn left', 'word(straight)', 'double white line', 'left turn box', 'word(sian)', 
              'speed reduction', 'word(slow)', 'stop line', 'word(zhuan)', 'word(dan)', 'word(go)', 
              'word(fast)', 'crosswalk', 'word(ji)', 'word(yi)', 'word(minus)', 'lane reduction', 
              'turn left or go straight', 'word(xiao)', 'turning line', 'word(speed)', 'word(yu)', 
              'pavement width transition marking', 'turn right', 'word(right)', 'word(car)', 
              'continuous lane lines', 'word(wood)', 'word(dao)', 'turn right or go straight', 
              'motorcycle waiting zone', 'turn left or right', 'word(chi)', 'speed limit(30)', 
              'word(use)', 'word(cha)', 'go straight or turn left or right', 'word(defend)', 'word(mountain)', 
              'word(left)', 'word(wan)', 'word(stop)', 'word(hua)', 'word(port)', 'left turn box(car)']

    colors = [(0,0,0),(255, 0, 0), (0, 255,0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), 
            (170, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), 
            (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), 
            (128, 255, 128), (128, 128, 255), (192, 192, 192), (128, 128, 128), (255, 165, 0), (210, 105, 30), 
            (0, 255, 127), (255, 20, 147), (255, 215, 0), (255, 99, 71), (255, 140, 0), (75, 0, 130), (255, 69, 0), 
            (0, 100, 0), (124, 252, 0), (173, 255, 47), (32, 178, 170), (95, 158, 160), (70, 130, 180), (65, 105, 225), 
            (72, 61, 139), (106, 90, 205), (123, 104, 238), (139, 0, 139), (148, 0, 211), (186, 85, 211), (165, 42, 42), 
            (178, 34, 34), (220, 20, 60), (255, 105, 180), (255, 182, 193), (255, 192, 203), (255, 228, 181), 
            (255, 235, 205), (255, 250, 205), (255, 255, 224), (200, 100, 50), (50, 200, 150), (55, 20, 147), 
            (255, 128, 64), (50, 50, 100), (200, 50, 200), (20, 200, 100), (100, 20, 200)]
    
    DatasetCatalog.register(name, lambda: CustomDataset(data_dir, split))
    MetadataCatalog.get(name).set(thing_classes=label_name, 
                                stuff_classes=label_name, 
                                thing_colors=colors,
                                stuff_colors=colors,
                                ignore_label=None, 
                                evaluator_type="custom_semseg")
