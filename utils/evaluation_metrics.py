import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score


class UncertaintyMetrics:
    """A class to compute and store uncertainty metrics."""
    def __init__(self):
        """Initializes empty lists to store metric data points."""
        self.auroc_data = []
        self.ece_data = []
        self.bbox_nll = []
        self.confidence_variance = []
        self.brier_scores = []
        self.entropy_scores = []

    @staticmethod
    def _calculate_ece(ece_data, n_bins=10):
        """
        Calculates the Expected Calibration Error (ECE).
        Bins all predictions by confidence and computes weighted avg. of |acc - conf|.
        """
        if not ece_data: return 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_n = len(ece_data)
        if total_n == 0: return 0.0

        for i in range(n_bins):
            bin_low = bin_boundaries[i]
            bin_high = bin_boundaries[i+1]

            # Find data points in this bin
            # Use <= bin_high for the last bin to include confidence = 1.0
            if i == n_bins - 1:
                in_bin = [d for d in ece_data if bin_low < d['confidence'] <= bin_high]
            else:
                in_bin = [d for d in ece_data if bin_low < d['confidence'] <= bin_high]

            if not in_bin: continue
            n_i = len(in_bin)
            bin_acc = np.mean([d['is_tp'] for d in in_bin])             # Accuracy in bin (proportion of TPs)
            bin_conf = np.mean([d['confidence'] for d in in_bin])       # Average confidence in bin
            ece += (n_i / total_n) * np.abs(bin_acc - bin_conf)
        return ece

    @staticmethod
    def _calculate_auarc(auarc_data, steps=11):
        """
        Calculates the Area Under the Accuracy-Rejection Curve.
        Sorts by uncertainty, rejects portions, and calculates accuracy (TP / (TP+FP))
        at each step. Returns the area under this curve.
        """
        if not auarc_data: return 0.0
        auarc_data.sort(key=lambda x: x['uncertainty'], reverse=True)
        total_n = len(auarc_data)
        accuracies = []
        rejection_rates = np.linspace(0, 1, steps)

        for rate in rejection_rates:
            n_to_reject = int(total_n * rate)
            if rate == 1.0:   # Handle edge case
                 n_to_reject = total_n
            
            kept_data = auarc_data[n_to_reject:]
            if not kept_data:
                accuracies.append(accuracies[-1] if accuracies else 0.0)
                continue

            num_tps = sum(1 for d in kept_data if d['label'] == 0)
            acc = num_tps / len(kept_data)
            accuracies.append(acc)
        return np.trapz(accuracies, rejection_rates)   # Using trapezoidal rule
    
    @staticmethod
    def calculate_shannon_entropy(probs):
        """
        Calculates Shannon Entropy for a full probability vector (Multi-class).
        High entropy = high uncertainty/confusion between classes.
        """
        epsilon = 1e-9
        probs_clipped = np.clip(probs, epsilon, 1.0)
        return -np.sum(probs_clipped * np.log2(probs_clipped))
    
    @staticmethod
    def calculate_binary_entropy(probs):
        """
        Calculates the total binary entropy for a vector of independent sigmoid probabilities.
        For each probability p_i, the binary entropy is -p_i*log2(p_i) - (1-p_i)*log2(1-p_i).
        This function sums this value over all class probabilities.
        """
        epsilon = 1e-7   # Prevent log(0) (undefined)
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        term1 = probs * np.log2(probs)
        term2 = (1 - probs) * np.log2(1 - probs)
        binary_entropies = -(term1 + term2)
        return np.mean(binary_entropies)

    @staticmethod
    def calculate_brier_score(probs, true_class_idx):
        """Calculates the Brier score for a given probability vector and true class."""
        num_classes = len(probs)
        true_label_one_hot = np.zeros(num_classes)
        if 0 <= true_class_idx < num_classes:
            true_label_one_hot[true_class_idx] = 1.0
        return np.mean((probs - true_label_one_hot)**2)

    def compute_bbox_nll(self, bbox_samples, true_bbox):
        if len(bbox_samples) < 2: return np.nan
        mean_bbox = np.mean(bbox_samples, axis=0)
        cov_bbox = np.cov(bbox_samples.T)
        cov_bbox += np.eye(4) * 1e-4 #1e-2 #4.0 #1e-2   #1e-3 #1e-6
        try:
            mvn = multivariate_normal(mean=mean_bbox, cov=cov_bbox, allow_singular=True)
            nll = -mvn.logpdf(true_bbox)
            if np.isinf(nll) or nll > 1e6:  
                return np.nan
        except (np.linalg.LinAlgError, ValueError):
            return np.nan
        return nll

    def update_tp_YD(self, cluster, matched_gt, coco_to_yolo_map, img_w, img_h, mean_score):
        """Calculates metrics for a True Positive cluster for YOLO / RTDETR."""
        norm = np.array([img_w, img_h, img_w, img_h])
        bbox_samples_pixel = np.array([det['bbox'] for det in cluster])
        true_bbox_pixel = np.array(matched_gt['bbox'])
        bbox_samples = bbox_samples_pixel / norm
        true_bbox = true_bbox_pixel / norm

        # Bbox NLL
        nll = self.compute_bbox_nll(bbox_samples, true_bbox)
        if not np.isnan(nll): self.bbox_nll.append(nll)
        
        # Classification Stability
        true_class_id_coco = matched_gt['category_id']
        
        # Confidence Variance
        conf_variance = np.var([det['score'] for det in cluster])
        self.confidence_variance.append(conf_variance)
        
        # For AUROC / AUARC
        prob_samples = np.array([det['full_probs'] for det in cluster])
        mean_probs = np.mean(prob_samples, axis=0)
        entropy = self.calculate_binary_entropy(mean_probs)
        self.entropy_scores.append(entropy)
        self.auroc_data.append({'uncertainty': entropy, 'label': 0})
        
        # Brier
        true_yolo_class_idx = coco_to_yolo_map.get(true_class_id_coco)
        if true_yolo_class_idx is not None:
            brier = self.calculate_brier_score(mean_probs, true_yolo_class_idx)
            self.brier_scores.append(brier)
        
        # We use the mean_score (objectness) as confidence and 1 as "correct" (is_tp)
        self.ece_data.append({'confidence': mean_score, 'is_tp': 1})

    def update_fp_YD(self, cluster, mean_score, img_w, img_h):
        """Calculates uncertainty for a False Positive cluster for YOLO / RTDETR."""
        bbox_samples = np.array([det['bbox'] for det in cluster])
        conf_variance = np.var([det['score'] for det in cluster])
        prob_samples = np.array([det['full_probs'] for det in cluster])
        mean_probs = np.mean(prob_samples, axis=0)
        entropy = self.calculate_binary_entropy(mean_probs)

        self.auroc_data.append({'uncertainty': entropy, 'label': 1})
        self.ece_data.append({'confidence': mean_score, 'is_tp': 0})

    def update_tp(self, cluster, matched_gt, img_w, img_h, mean_score):
        """Calculates metrics for a True Positive cluster for FasterRCNN."""
        norm = np.array([img_w, img_h, img_w, img_h])
        bbox_samples_pixel = np.array([det['bbox'] for det in cluster])
        true_bbox_pixel = np.array(matched_gt['bbox'])
        bbox_samples = bbox_samples_pixel / norm
        true_bbox = true_bbox_pixel / norm
        
        nll = self.compute_bbox_nll(bbox_samples, true_bbox)
        if not np.isnan(nll): self.bbox_nll.append(nll)
        
        conf_variance = np.var([det['score'] for det in cluster])
        self.confidence_variance.append(conf_variance)
        if 'all_scores' in cluster[0]:
            prob_vectors = np.array([det['all_scores'] for det in cluster])
            mean_probs_vector = np.mean(prob_vectors, axis=0)
            entropy = self.calculate_shannon_entropy(mean_probs_vector)
            
            true_class_id = matched_gt['category_id']
            brier = self.calculate_brier_score(mean_probs_vector, true_class_id)
            self.brier_scores.append(brier)
            
        else:
            entropy = self.calculate_binary_entropy(np.array([det['score'] for det in cluster]))
        
        self.entropy_scores.append(entropy)
        self.auroc_data.append({'uncertainty': entropy, 'label': 0}) 
        self.ece_data.append({'confidence': mean_score, 'is_tp': 1})

    def update_fp(self, cluster, mean_score, img_w, img_h):
        """Calculates uncertainty for a False Positive cluster for FasterRCNN."""
        norm = np.array([img_w, img_h, img_w, img_h])
        bbox_samples_pixel = np.array([det['bbox'] for det in cluster])
        bbox_samples = bbox_samples_pixel / norm
        conf_variance = np.var([det['score'] for det in cluster])
        self.confidence_variance.append(conf_variance)

        if 'all_scores' in cluster[0]:
            prob_vectors = np.array([det['all_scores'] for det in cluster])
            mean_probs_vector = np.mean(prob_vectors, axis=0)
            entropy = self.calculate_shannon_entropy(mean_probs_vector)
            true_class_id = 0 
            brier = self.calculate_brier_score(mean_probs_vector, true_class_id)
            self.brier_scores.append(brier)
        else:
            entropy = self.calculate_binary_entropy(np.array([det['score'] for det in cluster]))
        
        self.entropy_scores.append(entropy)
        self.auroc_data.append({'uncertainty': entropy, 'label': 1}) 
        self.ece_data.append({'confidence': mean_score, 'is_tp': 0})

    def summarize(self):
        """Returns a dictionary of the mean values and AUROC score."""
        summary = {
            "Mean Bbox NLL": np.nanmean(self.bbox_nll) if self.bbox_nll else 0,
            "Mean Confidence Variance": np.nanmean(self.confidence_variance) if self.confidence_variance else 0,
            "Mean Brier Score": np.nanmean(self.brier_scores) if self.brier_scores else 0,
            "Mean Entropy Score": np.nanmean(self.entropy_scores) if self.entropy_scores else 0,
        }
        
        if len(self.auroc_data) > 1 and len(set(d['label'] for d in self.auroc_data)) > 1:
            scores = [d['uncertainty'] for d in self.auroc_data]
            labels = [d['label'] for d in self.auroc_data]
            summary["Uncertainty AUROC"] = roc_auc_score(labels, scores)
        else:
            summary["Uncertainty AUROC"] = 0.0
        summary["ECE"] = self._calculate_ece(self.ece_data)
        summary["AUARC"] = self._calculate_auarc(list(self.auroc_data))
        return summary

