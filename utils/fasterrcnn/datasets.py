import torch
from torchvision.datasets import CocoDetection

class CocoDataset(CocoDetection):
    """
    Custom wrapper for the COCO dataset to make it compatible with PyTorch detection models.
    
    The standard torchvision.datasets.CocoDetection returns the raw COCO annotation dictionary.
    This class processes those annotations into the Tensor format required by Faster R-CNN.
    """
    def __init__(self, img_folder, ann_file, transform=None):
        """
        Args:
            img_folder (str): Path to the directory containing images.
            ann_file (str): Path to the JSON annotation file (e.g., instances_val2017.json).
            transform (callable, optional): A function/transform that takes in a PIL image 
                                            and returns a transformed version (usually a Tensor).
        """
        super().__init__(img_folder, ann_file)
        self.transform = transform

    def __getitem__(self, idx):
        """
        Loads an image and its annotations, processing them into a target dictionary.
        """
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        boxes, labels, area, iscrowd = [], [], [], []

        for obj in target:
            if obj.get('iscrowd', 0) == 1: continue
            if 'bbox' in obj:
                x, y, w, h = obj['bbox']   # COCO format is [x_min, y_min, width, height]
                if w > 0 and h > 0:  # Sanity Check
                    boxes.append([x, y, x + w, y + h])
                    labels.append(obj['category_id'])
                    area.append(obj['area'])
                    iscrowd.append(obj.get('iscrowd', 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
        }

        if self.transform is not None:
            img = self.transform(img)
        return img, target
