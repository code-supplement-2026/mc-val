import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import boxes as box_ops
from torchvision.models.resnet import Bottleneck
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from typing import List, Dict, Tuple, Optional
from timm.layers import DropPath
from timm.layers import DropBlock2d 


class RoIHeadsWithAllScores(RoIHeads):
    """
    Custom RoIHeads class that extends the standard torchvision RoIHeads.
    
    Main difference:
    The standard implementation only returns the highest scoring class label and score 
    per bounding box. This implementation returns the full probability vector ('all_scores')
    for every detected box.
    """
    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        
        pred_scores_vectors = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores_vectors.split(boxes_per_image, 0)
        detections: List[Dict[str, torch.Tensor]] = []
        
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            full_prob_vectors = scores.clone()
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            num_classes_no_bg = scores.shape[1]
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            inds_after_filtering = inds[keep]

            # Apply NMS
            keep_nms = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep_nms = keep_nms[: self.detections_per_img]
            boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

            final_flat_inds = inds_after_filtering[keep_nms]
            original_proposal_inds = final_flat_inds // num_classes_no_bg
            final_prob_vectors = full_prob_vectors[original_proposal_inds]

            detections.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "all_scores": final_prob_vectors
            })

        return detections

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # RoI Pooling and Forward pass
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            result = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

        return result, losses


def get_faster_rcnn_model(num_classes=91, regularization=None, drop_rate=0.2, use_custom_roi_heads=True, reg_stages: list = None):
    """
    Factory function to create a Faster R-CNN model with ResNet50-FPN backbone.
    - Can inject regularization (Dropout, DropBlock, Stochastic Depth) into the pretrained backbone.
    - Can replace standard RoIHeads with the custom version that returns all scores.
    
    Args:
        num_classes (int): Number of classes (including background).
        regularization (str, optional): 'dropout', 'droppath', or 'dropblock'.
        drop_rate (float): The maximum drop rate/probability.
        use_custom_roi_heads (bool): If True, uses RoIHeadsWithAllScores.
        reg_stages (list): List of ResNet stage names to apply regularization to (e.g., ['layer3', 'layer4']).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    if regularization:
        if reg_stages is None:
            reg_stages = ['layer1', 'layer2', 'layer3', 'layer4']
        
        BlockClass, kwargs_key = None, ""
        if regularization == 'MCD':
            BlockClass, kwargs_key = BottleneckDropout, 'dropout_rate'
        elif regularization == 'MCSD':
            BlockClass, kwargs_key = BottleneckDropPath, 'drop_path_rate'
        elif regularization == 'MCDB':
            BlockClass, kwargs_key = BottleneckDropBlock, 'drop_prob'
        else:
            raise ValueError(f"Unknown regularization type: {regularization}")

        blocks_to_replace = []
        for stage_name in reg_stages:
            stage_module = getattr(model.backbone.body, stage_name, None)
            if stage_module is None:
                print(f"WARNING: Stage '{stage_name}' not found in backbone. Skipping.")
                continue
            for block in stage_module:
                if isinstance(block, Bottleneck):
                    blocks_to_replace.append(block)
        
        total_blocks = len(blocks_to_replace)
        if total_blocks == 0:
            print(f"WARNING: No Bottleneck blocks found in stages {reg_stages}.")
        dp_rates = torch.linspace(0, drop_rate, total_blocks).tolist() if total_blocks > 0 else []
        rate_idx = 0

        # In-place replacement of blocks
        for stage_name in reg_stages:
            stage_module = getattr(model.backbone.body, stage_name, None)
            if stage_module is None:
                continue 
                
            for i in range(len(stage_module)):
                child = stage_module[i]
                if isinstance(child, Bottleneck):
                    # Extract configuration from existing block
                    inplanes, planes = child.conv1.in_channels, child.conv3.out_channels // Bottleneck.expansion
                    stride, downsample = child.stride, child.downsample
                    groups, dilation = child.conv2.groups, child.conv2.dilation[0]

                    block_kwargs = {kwargs_key: dp_rates[rate_idx]}
                    if regularization == 'MCDB':
                        block_kwargs['block_size'] = 7

                    new_bottleneck = BlockClass(
                        inplanes=inplanes, planes=planes, stride=stride,
                        downsample=downsample, groups=groups, dilation=dilation,
                        norm_layer=type(child.bn1), **block_kwargs
                    )
                    
                    new_bottleneck.load_state_dict(child.state_dict(), strict=False)
                    stage_module[i] = new_bottleneck
                    rate_idx += 1
        print(f"INFO: Successfully replaced {rate_idx} blocks.")

    # Replace RoIHeads if requested
    if use_custom_roi_heads:
        print("INFO: Replacing standard RoIHeads with RoIHeadsWithAllScores.")
        original_heads = model.roi_heads
        
        new_roi_heads = RoIHeadsWithAllScores(
            box_roi_pool=original_heads.box_roi_pool,
            box_head=original_heads.box_head,
            box_predictor=original_heads.box_predictor,
            fg_iou_thresh=original_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=original_heads.proposal_matcher.low_threshold,
            batch_size_per_image=original_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=original_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=original_heads.box_coder.weights,
            score_thresh=original_heads.score_thresh,
            nms_thresh=original_heads.nms_thresh,
            detections_per_img=original_heads.detections_per_img,
        )
        model.roi_heads = new_roi_heads

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class BottleneckDropout(nn.Module):
    """
    ResNet Bottleneck block with standard 2D Dropout.
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 dropout_rate=0.0):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # --- Standard ResNet Layers ---
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # --- Custom Regularization ---
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return self.dropout(out)
    

class BottleneckDropPath(nn.Module):
    """
    ResNet Bottleneck block with DropPath (Stochastic Depth).
    Randomly drops the entire residual path based on the probability.
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 drop_path_rate=0.0):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # timm's DropPath for Stochastic Depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Apply DropPath
        out = self.drop_path(out)
        out = out + identity
        out = self.relu(out)

        return out
    
    
class BottleneckDropBlock(nn.Module):
    """
    ResNet Bottleneck block with DropBlock.
    Drops contiguous regions of feature maps (spatial dropout).
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 drop_prob=0.0, block_size=7):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # DropBlock
        self.drop_block = DropBlock2d(drop_prob=drop_prob, block_size=block_size) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Apply DropBlock
        out = self.drop_block(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
