# Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning

This repository contains the implementations of the uncertainty estimation methods Monte Carlo Dropout (MCD), Monte Carlo DropBlock (MCDB) and Monte Carlo Stochastic Depth (MCSD).  
The Monte Carlo methods are made available for the models:  
- Faster R-CNN      (Torchvision)
- YOLOv8x           (Ultralytics)
- RT-DETRx          (Ultralytics)

Full dependencies are listed in `requirements.txt`.

## Installation

```bash
# Clone the repository
git clone https://github.com/code-supplement-26/mc-val.git
cd mc-val

# Install requirements
pip install -r requirements.txt
```

## Usage

We provide a fork of the ultralytics repository (https://github.com/ultralytics/ultralytics) in which we adapted some files (see Supplementary Material).  

As two annotation files are too large for github, we provide them via [Zenodo](https://zenodo.org/records/18743681).  
The two .json files supplied via Zenodo, are to be copied into /data/COCO/ultralytics_format/annotations.

For datasets we only provide the structure, as the full datasets are available online:
- COCO: https://cocodataset.org
- COCO-O: https://github.com/alibaba/easyrobust/tree/main/benchmarks/coco_o

More information on datasets is available in the README.md in the /data subdirectory.

## Repository Structure

```
mc-val/
├── run_inference_FasterRCNN.py        # Main implementation for Faster R-CNN
├── run_inference_YOLO_or_RTDETR.py    # Main implementation for YOLO and RT-DETR
├── utils/                             # Helper functions and adaptations of Torchvision and Ultralytics
│   ├── evaluation_metrics.py          # Uncertainty metrics
│   ├── helper_functions.py            # Helper functions for the main scripts
│   ├── fasterrcnn/                    # Adaptations to Torchvisions Faster R-CNN
│   └── ultralytics/                   # Ultralytics adapted files
├── data/                              # Datasets. Detailed information in the included README.md
│   └── ...
└── ...
```
