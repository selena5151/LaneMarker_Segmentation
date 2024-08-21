# Lane marker segmentation based on Transformer

Lane marker sementic segmentation base on [Lawin Transformer](https://github.com/yan-hao-tian/VW/tree/master/MaskFormer) [[paper](https://arxiv.org/abs/2201.01615)].


## Installation

Creating a new environment with Anaconda.
```
cd lawin
conda env create -f lawin.yaml --name lawin
conda activate lawin
```
Install [Detectron2](https://github.com/facebookresearch/detectron2?tab=readme-ov-file) and [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf)
```
cd ..
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Datasets

Lane Marker Dataset is prepared by [Datasetcreator](https://github.com/Eashwar93/Datasetcreator).

More styles of dataset: See [Preparing Datasets for MaskFormer](datasets/README.md).

## Train

More Utilization: See [Getting Started with MaskFormer](GETTING_STARTED.md). 

Swin-Tiny
```
cd lawin
python train_net.py --config-file configs/roadmarking/swin/lawin/lawin_maskformer_swin_tiny_bs16_90k.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.001 OUTPUT_DIR output/lawintiny_batch8_lanemarker
```
Swin-Small
```
cd lawin
python train_net.py --config-file configs/roadmarking/swin/lawin/lawin_maskformer_swin_small_bs16_90k.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.001 OUTPUT_DIR output/lawinsmall_batch8_lanemarker
```

Swin-Base
```
cd lawin
python train_net.py --config-file configs/roadmarking/swin/lawin/lawin_maskformer_swin_base_bs16_90k.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.001 OUTPUT_DIR output/lawinbase_batch8_lanemarker
```

## Evaluation
```
cd lawin
python train_net.py --eval-only --config-file path/to/config --num-gpus NGPUS OUTPUT_DIR path/to/output MODEL.WEIGHTS path/to/weight
```
## Demo
More information is explained in [demo/README.md](demo/README.md).
```
cd lawin
python demo/demo.py --config-file path/to/config --input path/to/input_image/*.jpg  --output path/to/output/ --opts path/to/weight
```