import os
import sys
import pandas as pd
import torch.nn as nn

ultralytics_path = "./utils/ultralytics"
sys.path.insert(0, os.path.abspath(ultralytics_path))
from ultralytics.nn.modules.block import Bottleneck, DOBottleneck, SDBottleneck, DBBottleneck, HGBlock


class BottleneckController:
    """
    Manages the replacement of standard Bottlenecks with MC-Dropout variants (DO, DB, or SD).
    """
    def __init__(self, indices_to_replace, method, decay_mode="same", reg_rate=0.01):
        """
        Args:
            indices_to_replace (set): Indices of bottlenecks to replace.
            method (str): "MCD", "MCDB", or "MCSD".
            decay_mode (str): "same" (constant rate) or "linear" (linearly increasing).
            drop_rate (float): The base probability for dropout/dropblock/droppath.
        """
        self.indices_to_replace = set(indices_to_replace)
        self.counter = 0
        self.method = method
        self.decay_mode = decay_mode

        # Base rates (will be max if linear decay)
        self.dropout_rate = reg_rate
        self.droppath_rate = reg_rate
        self.dropblock_rate = reg_rate
            
        # Properties for linear decay
        self.sorted_indices = sorted(list(self.indices_to_replace))
        self.L = len(self.sorted_indices)
        self.index_to_local_l = {global_idx: l for l, global_idx in enumerate(self.sorted_indices)}

    def get_class(self):
        """
        Determines and returns the appropriate Bottleneck class constructor 
        (standard or Bayesian) for the current layer index.
        """
        is_target = self.counter in self.indices_to_replace
        current_global_index = self.counter
        self.counter += 1
        if is_target:
            if self.method == "MCD":
                print(f"Instance {self.counter-1}: Replacing with DOBottleneck")
                return lambda *args, **kwargs: DOBottleneck(*args, **kwargs, dropout_rate=self.dropout_rate)
            elif self.method == "MCDB":
                print(f"Instance {self.counter-1}: Replacing with DBBottleneck")
                return lambda *args, **kwargs: DBBottleneck(*args, **kwargs, dropblock_rate=self.dropblock_rate)
            elif self.method == "MCSD":
                current_droppath_rate = self.droppath_rate
                if self.decay_mode == "linear" and self.L > 0:
                    l = self.index_to_local_l.get(current_global_index)
                    if l is not None:   # Linear decay
                        current_droppath_rate = ( (l + 1) / self.L ) * self.droppath_rate
                        print(f"Instance {current_global_index}: Replacing with SDBottleneck (Linear Rate: {current_droppath_rate:.4f} - l={l}, L={self.L})")
                    else:   # Fallback
                        print(f"Instance {current_global_index}: Replacing with SDBottleneck (Rate: {self.droppath_rate}) - Fallback")
                elif self.decay_mode == "same":
                    print(f"Instance {current_global_index}: Replacing with SDBottleneck (Same Rate: {self.droppath_rate})")
                return lambda *args, **kwargs: SDBottleneck(*args, **kwargs, droppath_rate=current_droppath_rate)
        else:
            return Bottleneck


class HGBlockController:
    """
    Determine regularisation parameters for HGBlocks. Modify instances after their creation.
    """
    def __init__(self, indices_to_replace, method, reg_rate, decay_mode="same"):
        """
        Args:
            indices_to_replace (set): Set of global HGBlock indices to modify.
            method (str): "MCD", "MCDB", or "MCSD".
            reg_rate (float): Base/max registration rate.
            decay_mode (str): "same" or "linear".
        """
        self.indices_to_replace = set(indices_to_replace)
        self.method = method
        self.base_reg_rate = reg_rate
        self.decay_mode = decay_mode
        self.detr_method_dict = {"MCD": "DO", "MCDB": "DB", "MCSD": "SD"}
        self.reg_type = self.detr_method_dict.get(self.method)

        # Properties for linear decay
        self.sorted_indices = sorted(list(self.indices_to_replace))
        self.L = len(self.sorted_indices)
        self.index_to_local_l = {global_idx: l for l, global_idx in enumerate(self.sorted_indices)}

    def get_params_for_index(self, global_index):
        """
        Calculates the (reg_type, reg_rate) for a given global index.
        Returns (None, 0.0) if this index should not be modified.
        """
        if global_index not in self.indices_to_replace or not self.reg_type:
            return None, 0.0
        current_reg_rate = self.base_reg_rate

        if self.method == "MCSD" and self.decay_mode == "linear" and self.L > 0:
            l = self.index_to_local_l.get(global_index)
            if l is not None:
                current_reg_rate = ( (l + 1) / self.L ) * self.base_reg_rate
                print(f"Index {global_index}: Setting HGBlock (Linear Rate: {current_reg_rate:.4f} l={l}, L={self.L})")
            else:   # Fallback
                print(f"Index {global_index}: Setting HGBlock (Same Rate: {current_reg_rate}) - Fallback")
        else:
            print(f"Index {global_index}: Setting HGBlock (Same Rate: {current_reg_rate})")
        return self.reg_type, current_reg_rate


def force_dropout(model):
    """Set all dropout layers are to training mode."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, DBBottleneck, DOBottleneck, SDBottleneck, HGBlock)):
            m.train()

def activate_mc_sampling(model):
    """
    Activates stochastic layers (Dropout, DropPath, DropBlock) during inference for FasterRCNN."""
    for m in model.modules():
        # Check for standard PyTorch layers and custom timm layers
        if m.__class__.__name__ in ['Dropout', 'Dropout2d', 'DropPath', 'DropBlock2D']:
            m.train()

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format."""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def save_results(results_list, csv_filename):
    """Saves the list of results dictionaries to a CSV file."""
    df = pd.DataFrame(results_list)
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved results to {csv_filename}")

def log_results(model, method, mc, dr, data, conf_thresh, adap_layers, stats, uncertainty_summary, csv_file):
    """Prints metrics, and concats results."""
    print(f"Evaluating on {os.path.basename(data)}")
    print("\n--- COCO Evaluation Metrics (mAP) ---")
    print(f"mAP@.5:.95: {stats[0]:.4f}")
    print(f"mAP@.5:    {stats[1]:.4f}")
    print("\n--- Uncertainty Metrics ---")
    results = []
    auroc_score = 0.0
    ece_score = 0.0
    auarc_score = 0.0
    
    if uncertainty_summary:
        auroc_score = uncertainty_summary.pop("Uncertainty AUROC", 0.0)
        ece_score = uncertainty_summary.pop("ECE", 0.0)
        auarc_score = uncertainty_summary.pop("AUARC", 0.0)

        print("Metrics for True Positives:")
        for key, value in uncertainty_summary.items():
            print(f"  {key}: {value:.4f}")
        print("\nMetrics for All Detections (TP vs FP):")
        print(f"  Uncertainty AUROC: {auroc_score:.4f}")
        print(f"  Expected Calibration Error (ECE): {ece_score:.4f}")
        print(f"  Accuracy-Rejection Curve (AUARC): {auarc_score:.4f}")
    else:
        print("No matched detections to calculate uncertainty statistics.")

    results.append({
        "method": method,
        "mc_samples": mc,
        "drop_rate": dr,
        "adapted_layers": str(adap_layers) if adap_layers is not None else "N/A",
        "dataset": os.path.basename(data),
        "conf_th": conf_thresh,
        "mAP50-95": stats[0],
        "mAP50": stats[1],
        "Mean Bbox NLL": uncertainty_summary.pop("Mean Bbox NLL", 0.0),
        "Mean Confidence Variance": uncertainty_summary.pop("Mean Confidence Variance", 0.0),
        "Mean Brier Score": uncertainty_summary.pop("Mean Brier Score", 0.0),
        "Mean Entropy Score": uncertainty_summary.pop("Mean Entropy Score", 0.0),
        "AUROC (entropy)": auroc_score,
        "Expected Calibration Error": ece_score,
        "AUARC": auarc_score
    })

    save_results(results, csv_file)

