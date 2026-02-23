import os
import torch
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from utils.fasterrcnn.datasets import CocoDataset
from utils.fasterrcnn.faster_rcnn import get_faster_rcnn_model
from utils.evaluation_metrics import UncertaintyMetrics
from utils.helper_functions import activate_mc_sampling, calculate_iou


def collate_fn(batch):
    """Custom collate function to handle images of different sizes."""
    return tuple(zip(*batch))

def run_evaluation(reg_type, saved_model_path, drop_rate, mc_samples, confidence_threshold, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Evaluation for: {reg_type} ---")
    print(f"Checkpoint: {saved_model_path}")
    print(f"Parameters: drop_rate={drop_rate}, mc_samples={mc_samples}, conf_thresh={confidence_threshold}")

    val_img_folder = os.path.join(data_path, "val2017")
    val_ann_file = os.path.join(data_path, "annotations/instances_val2017.json")
    val_dataset = CocoDataset(val_img_folder, val_ann_file, T.ToTensor())
    coco_gt = COCO(val_ann_file)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

    num_classes = 91
    model = get_faster_rcnn_model(
        num_classes=num_classes,
        regularization=reg_type if reg_type != 'baseline' else None,
        drop_rate=drop_rate,
        use_custom_roi_heads=True,    # Ensures return of full probability vectors
    )
    
    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    uncertainty_metrics = UncertaintyMetrics()
    coco_results_for_map = []
    if mc_samples > 1 and model != 'baseline':
        activate_mc_sampling(model)

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Inference"):
            imgs = [img.to(device) for img in imgs]
            target = targets[0]
            
            mc_detections = []
            for _ in range(mc_samples):   # Sampling loop
                outputs = model(imgs)[0]
                
                # PyTorch format (x1, y1, x2, y2) to COCO format (x, y, w, h)
                boxes_xyxy = outputs['boxes'].cpu()
                boxes_xywh = boxes_xyxy.clone()
                boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0] # Width
                boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1] # Height
                has_all_scores = 'all_scores' in outputs
                
                for i in range(len(outputs['scores'])):
                    detection_data = {
                        'bbox': boxes_xywh[i].tolist(),
                        'score': outputs['scores'][i].item(),
                        'category_id': outputs['labels'][i].item()
                    }
                    if has_all_scores:
                        detection_data['all_scores'] = outputs['all_scores'][i].cpu().numpy()
                    mc_detections.append(detection_data)
            if not mc_detections: continue

            # Clustering (greedy)
            clusters = []
            mc_detections.sort(key=lambda x: x['score'], reverse=True)
        
            while mc_detections:
                current = mc_detections.pop(0)
                cluster = [current]
                remaining = []
                
                for det in mc_detections:
                    if det['category_id'] == current['category_id'] and calculate_iou(det['bbox'], current['bbox']) > 0.5:
                        cluster.append(det)
                    else:
                        remaining.append(det)
                clusters.append(cluster)
                mc_detections = remaining
            
            gt_annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=target['image_id'].item()))
            used_gt_indices = set() # Track already matched GT objects

            for cluster in clusters:
                mean_score = np.mean([d['score'] for d in cluster])
                if mean_score < confidence_threshold: continue
                mean_box = np.mean([d['bbox'] for d in cluster], axis=0).tolist()
                best_iou, best_gt_idx = 0.5, -1
                
                for gt_idx, gt in enumerate(gt_annotations):
                    if gt_idx in used_gt_indices or gt['category_id'] != cluster[0]['category_id']: continue

                    iou = calculate_iou(mean_box, gt['bbox'])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx
                img_h, img_w = imgs[0].shape[1], imgs[0].shape[2]
                
                if best_gt_idx != -1:
                    used_gt_indices.add(best_gt_idx)
                    uncertainty_metrics.update_tp(cluster, gt_annotations[best_gt_idx], img_w, img_h, mean_score)
                else:
                    uncertainty_metrics.update_fp(cluster, mean_score, img_w, img_h)

                coco_results_for_map.append({
                    'image_id': target['image_id'].item(),
                    'category_id': cluster[0]['category_id'],
                    'bbox': mean_box,
                    'score': mean_score
                })

    stats = [0.0] * 12
    if coco_results_for_map:
        coco_dt = coco_gt.loadRes(coco_results_for_map)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats

    final_uncertainty_summary = uncertainty_metrics.summarize()

    # Compile all results
    results = {
        "model": model,
        "mc_samples": mc_samples,
        "drop_rate": drop_rate if model != 'baseline' else 0,
        "conf_thresh": confidence_threshold,
        "mAP50-95": stats[0],
        "mAP50": stats[1],
        **final_uncertainty_summary
    }
    return results


if __name__ == "__main__":
    method = 'MCD'      # Options: 'MCD', 'MCDB', 'MCSD'
    n_samples = 20
    drop_rate = 0.1
    data = './data/COCO/coco_format'
    conf_th = 0.4
    
    model_dict = {'MCD': './models/FasterRCNN_MCD.pth', 'MCDB': './models/FasterRCNN_MCDB.pth', 'MCSD': './models/FasterRCNN_MCSD.pth'}
    saved_model_path = model_dict.get(method)
    
    result = run_evaluation(method, saved_model_path, drop_rate, n_samples, conf_th, data)
        
    print("\n--- Evaluation Results ---")
        
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("Finished process.")

