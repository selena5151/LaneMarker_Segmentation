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
        self.singleword_dir = os.path.join(data_dir, "singleWord_label", split)
        # self.singleword_dir = os.path.join(data_dir, "bike_label", split)
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
                label_file = os.path.join(self.singleword_dir, f"{image_id}_id.png")
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


def register_singleword_dataset(name, data_dir, split):

    # 28 classes
    label_name = ['unlabeled', 'crossover crosswalk', 'go straight', 'bicycle lane', 'double yellow line', 'broken lane line',
    'giveaway', 'guide sign', 'cross hatch', 'obstacle', 'turn left', 'double white line', 'left turn box', 'speed reduction',
    'stop line', 'crosswalk', 'lane reduction', 'turn left or go straight', 'turning line', 'pavement width transition marking',
    'turn right', 'continuous lane lines', 'turn right or go straight', 'motorcycle waiting zone', 'turn left or right',
    'go straight or turn left or right', 'left turn box(car)','word']

    colors = [(0, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 128), (128, 128, 0), (128, 0, 128), (255, 128, 0),
        (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 128, 255), (192, 192, 192), (255, 165, 0), (0, 255, 127),
        (75, 0, 130), (173, 255, 47), (32, 178, 170), (70, 130, 180), (106, 90, 205), (123, 104, 238), (186, 85, 211),
        (220, 20, 60), (255, 105, 180), (255, 182, 193), (255, 255, 224), (100, 20, 200),(0, 0, 255)]

    
    DatasetCatalog.register(name, lambda: CustomDataset(data_dir, split))
    MetadataCatalog.get(name).set(thing_classes=label_name, 
                                stuff_classes=label_name, 
                                thing_colors=colors,
                                stuff_colors=colors,
                                ignore_label=255, evaluator_type="custom_semseg")
                                #evaluator_type="binary_iou")
