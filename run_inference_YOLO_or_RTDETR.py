import os
import sys
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.evaluation_metrics import UncertaintyMetrics
from utils.helper_functions import BottleneckController, HGBlockController, force_dropout, calculate_iou, log_results, save_results

ultralytics_path = "./utils/ultralytics"
sys.path.insert(0, os.path.abspath(ultralytics_path))
from ultralytics import YOLO
from ultralytics.nn.modules import block


def validate_coco(model, method, data_path, conf_threshold, split='val', batch_size=64, num_samples=3):
    """
    Runs test on COCO dataset using Monte Carlo methods (MCD/MCDB/MCSD) and calculates evaluation metrics.

    Args:
        model (model): The loaded Ultralytics model.
        method (str): Inference method ('MCD', 'MCDB', or 'MCSD').
        data_path (str): Path to COCO dataset root.
        conf_threshold (float): Confidence threshold for detections.
        split (str): Dataset split to use (default 'val').
        batch_size (int): Batch size for inference loops.
        num_samples (int): Number of forward passes for uncertainty estimation.

    Returns:
        tuple: (coco_eval.stats, uncertainty_summary) - mAP stats list and uncertainty metrics dict.
    """
    
    ann_file = f"{data_path}/annotations/instances_{split}2017.json"
    coco = COCO(ann_file)
    img_dir = f"{data_path}/images/{split}2017"
    img_ids = coco.getImgIds()
    img_info = coco.loadImgs(img_ids)
    img_paths = [os.path.join(img_dir, img['file_name']) for img in img_info]
    
    img_id_lookup = {img['file_name']: img['id'] for img in img_info}
    coco_cat_ids = sorted(coco.getCatIds())
    yolo_to_coco_map = {i: cat_id for i, cat_id in enumerate(coco_cat_ids)}
    
    coco_to_yolo_map = {v: k for k, v in yolo_to_coco_map.items()}
    all_gts = {img_id: coco.loadAnns(coco.getAnnIds(imgIds=img_id)) for img_id in img_ids}
    uncertainty_metrics = UncertaintyMetrics()
    coco_results_for_map = []


    for i in tqdm(range(0, len(img_paths), batch_size), desc=f"Running {method} Inference"):
        batch_paths = img_paths[i:i + batch_size]
        batch_mc_detections = {path: [] for path in batch_paths}


        for sample_idx in range(num_samples):
            if method in ["MCD"]: force_dropout(model)   # Activate Dropout if MCD
            results = model(batch_paths, conf=conf_threshold, verbose=False)
            batch_full_probs = model.predictor.full_probs

            for res_idx, res in enumerate(results):
                image_probs = batch_full_probs[res_idx]  # Full prob vector of current image
                file_name = os.path.basename(res.path)
                image_id = img_id_lookup.get(file_name)
                if image_id is None: continue

                for box_idx, box in enumerate(res.boxes):
                    detection_full_probs = image_probs[box_idx].cpu().numpy()
                    cls_idx = int(box.cls.item())

                    if cls_idx not in yolo_to_coco_map: continue

                    x_c, y_c, w, h = box.xywh[0].cpu().tolist()

                    detection = {
                        'image_id': image_id,
                        'category_id': yolo_to_coco_map[cls_idx],
                        'bbox': [x_c - w / 2, y_c - h / 2, w, h],
                        'score': float(box.conf),
                        'sample_idx': sample_idx,
                        'full_probs': detection_full_probs
                    }

                    batch_mc_detections[res.path].append(detection)


        for path, all_detections in batch_mc_detections.items():
            if not all_detections: continue
            clusters = []
            all_detections.sort(key=lambda x: x['score'], reverse=True)
            
            while all_detections:  # Clustering logic
                current = all_detections.pop(0)
                cluster = [current]
                remaining = []
                for det in all_detections:
                    if det['category_id'] == current['category_id'] and calculate_iou(det['bbox'], current['bbox']) > 0.5:
                        cluster.append(det)
                    else:
                        remaining.append(det)
                clusters.append(cluster)
                all_detections = remaining

            current_image_id = img_id_lookup[os.path.basename(path)]
            gt_annotations = all_gts[current_image_id]
            used_gt_indices = set()
            img_meta = coco.loadImgs(current_image_id)[0]
            img_w, img_h = img_meta['width'], img_meta['height']


            for cluster in clusters:
                mean_box = np.mean([d['bbox'] for d in cluster], axis=0).tolist()
                mean_score = np.mean([d['score'] for d in cluster])
                best_iou = 0.5
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_annotations):
                    if gt_idx in used_gt_indices: continue
                    if gt['category_id'] != cluster[0]['category_id']: continue

                    iou = calculate_iou(mean_box, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx != -1:
                    used_gt_indices.add(best_gt_idx)
                    matched_gt = gt_annotations[best_gt_idx]
                    uncertainty_metrics.update_tp_YD(cluster, matched_gt, coco_to_yolo_map, img_w, img_h, mean_score)
                else:
                    uncertainty_metrics.update_fp_YD(cluster, mean_score, img_w, img_h)

                coco_results_for_map.append({
                    'image_id': current_image_id,
                    'category_id': cluster[0]['category_id'],
                    'bbox': mean_box,
                    'score': mean_score
                })

    if not coco_results_for_map:
        print("No predictions to evaluate for mAP.")
        final_uncertainty_summary = uncertainty_metrics.summarize()
        return [0.0] * 12, final_uncertainty_summary

    coco_dt = coco.loadRes(coco_results_for_map)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats, uncertainty_metrics.summarize()


if __name__ == '__main__':
    model_to_run = 'RT-DETR-x'      # Options: 'RT-DETR-x', 'YOLOv8x'
    method = 'MCSD'                 # Options: 'MCD', 'MCDB', 'MCSD'
    n_samples = 20
    drop_rate = 0.2
    data = './data/COCO/ultralytics_format'
    csv_file = "results_AblationNew.csv"
    conf_th = 0.5
    adap_bn = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
    adap_hg = range(12)
    SD_decay_mode = "linear"        # Or "same" for no decay. Only applies to MCSD

    print(f"\nRunning on {model_to_run}: {method}, {n_samples}, {drop_rate}, {data}, {conf_th}, {adap_hg}")

    if model_to_run == 'YOLOv8x':
        _OriginalBottleneck = block.Bottleneck
        controller = BottleneckController(adap_bn, method, decay_mode=SD_decay_mode, reg_rate=drop_rate)

        class ConditionalBottleneck(nn.Module):
            def __new__(cls, *args, **kwargs):
                return controller.get_class()(*args, **kwargs)

        block.Bottleneck = ConditionalBottleneck
        model = YOLO('yolov8x.yaml')                # Build model
        block.Bottleneck = _OriginalBottleneck
        yolo_model_dict = {'MCD': './models/YOLO_MCD.pt', 'MCDB': './models/YOLO_MCDB.pt', 'MCSD': './models/YOLO_MCSD.pt'}
        model.load(yolo_model_dict.get(method))     # Load weights
        print(f"  Total Bottleneck instances processed: {controller.counter}")
    
    elif model_to_run == 'RT-DETR-x':
        detr_model_dict = {'MCD': './models/RTDETR_MCD.pt', 'MCDB': './models/RTDETR_MCDB.pt', 'MCSD': './models/RTDETR_MCSD.pt'}
        model = YOLO(detr_model_dict.get(method))

        # Initialize default
        for m in model.model.modules():
            if isinstance(m, block.HGBlock):
                if not hasattr(m, 'reg_type'):
                    m.reg_type = 'none'
                    m.reg_rate = 0.0

        controller = HGBlockController(adap_hg, method, reg_rate=drop_rate, decay_mode=SD_decay_mode)
        hg_block_modules = [m for m in model.model.modules() if isinstance(m, block.HGBlock)]
        print(f"  Found {len(hg_block_modules)} HGBlock instances to process.")
        
        for global_index, module in enumerate(hg_block_modules):
            reg_type, reg_rate = controller.get_params_for_index(global_index)
            if reg_type: module.reg_type, module.reg_rate = reg_type, reg_rate

    stats, uncertainty_summary = validate_coco(model, method, data, conf_th, num_samples=n_samples)
    log_results(model, method, n_samples, drop_rate, data, conf_th, adap_hg, stats, uncertainty_summary, csv_file)
    print("Finished process.")

