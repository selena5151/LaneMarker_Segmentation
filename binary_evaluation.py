# 只計算前景與背景的iou
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


class BinaryIoUEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

    def reset(self):
        self._conf_matrix = np.zeros((2, 2), dtype=np.int64)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            output = output['sem_seg'].argmax(dim=0).to(self._cpu_device)
            pred = (np.array(output, dtype=int) != 0).astype(int)  # 將非0類別視為1（前景），0為背景

            gt_filename = self.input_file_to_gt_file[input['file_name']]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            gt = (gt != 0).astype(int)  # 將標籤為0的類別視為背景，其他為前景

            # 更新混淆矩陣
            self._conf_matrix += np.bincount(
                2 * pred.reshape(-1) + gt.reshape(-1), 
                minlength=self._conf_matrix.size
            ).reshape(self._conf_matrix.shape)

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            if not is_main_process():
                return

            self._conf_matrix = np.sum(conf_matrix_list, axis=0)

        tp = self._conf_matrix[1, 1]
        fp = self._conf_matrix[0, 1]
        fn = self._conf_matrix[1, 0]
        tn = self._conf_matrix[0, 0]

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0

        results = {
            "IoU": 100 * iou
        }

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "binary_iou_evaluation.json")
            with open(file_path, "w") as f:
                f.write(json.dumps(results))

        return {"binary_iou": results}
