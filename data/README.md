# Datasets

This directory contains the COCO and COCO-O datasets in the respective formats for use with Ultralytics or Torchvision.  
We do not provide the full datasets, as they are available on https://cocodataset.org (COCO) and https://github.com/alibaba/easyrobust/tree/main/benchmarks/coco_o (COCO-O).  
Specific formatting is shown in the structure, where for "ultralytics_format" directories only the images from both COCO and COCO-O have to be supplied.  


## Structure of Subdirectories

Structure of the COCO (ID) dataset:

```
COCO/
├── coco_format/                            # COCO dataset in the conventional format, as available at https://cocodataset.org/#download
│   ├── annotations/
│   │   └── ...
│   ├── images/
│   │   └── ...
│   ├── labels/
│   │   └── ...
│   ├── train2017.txt
│   └── ...
└── ultralytics_format/                     # COCO dataset reformatted. We provide everything except the image files
    ├── annotations/
    │   ├── captions_train2017.json
    │   └── ...
    ├── images/
    │   ├── test2017/
    │   │   ├── 000000000001.jpg
    │   │   └── ...
    │   ├── train2017/
    │   │   └── ...
    │   └── val2017/
    │       └── ...
    └── labels/
        └── annotations/
```

For COCO-O (distribution shifted data) one subdirectory layer is added:

```
COCO-O/
├── cartoon/                            # COCO-O dataset with domain shift "cartoon"
│   ├── coco_format/
│   │   ├── annotations/
│   │   └── ...
│   └── ultralytics_format/
│       ├── annotations/
│       └── ...
├── handmake/                            # COCO-O dataset with domain shift "cartoon"
├── painting/                            # COCO-O dataset with domain shift "cartoon"
├── sketch/                            # COCO-O dataset with domain shift "cartoon"
├── tattoo/                            # COCO-O dataset with domain shift "cartoon"
└── weather/                            # COCO-O dataset with domain shift "cartoon"
```
