# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Loss functions with anatomical region adaptation for DADA framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


def classify_anatomical_region(feature_map, bbox):
    """
    Classify the anatomical region type to guide region-adaptive perturbation.
    Region types:
    0 = lumen (circular structures)
    1 = folds (linear structures)
    2 = specular highlights (bright spots)
    3 = flat mucosa (uniform texture)
    """
    # Extract region from feature map
    x1, y1, x2, y2 = bbox
    region = feature_map[:, :, int(y1):int(y2), int(x1):int(x2)]
    
    if region.numel() == 0:
        return 3  # Default to flat mucosa
    
    # Calculate region statistics
    mean_val = region.mean()
    std_val = region.std()
    aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)
    
    # Simple heuristic classification
    if std_val < 0.1 and mean_val > 0.7:  # Uniform bright region
        return 2  # specular highlight
    elif aspect_ratio > 2.0 or aspect_ratio < 0.5:  # Elongated structure
        return 1  # fold
    elif 0.8 < aspect_ratio < 1.2 and std_val > 0.3:  # Circular with texture
        return 0  # lumen
    else:
        return 3  # flat mucosa


def get_region_adaptive_alpha(region_type, base_alpha=0.003):
    """
    Get adaptive perturbation step size based on anatomical region type.
    Stronger perturbations for regions prone to false positives.
    """
    region_alphas = {
        0: 0.004,   # Lumen - needs stronger perturbation
        1: 0.0035,  # Folds - moderate perturbation
        2: 0.0025,  # Specular highlights - subtle perturbation
        3: 0.002    # Flat mucosa - minimal perturbation
    }
    return region_alphas.get(region_type, base_alpha)


class AnatomicalRegionLoss(nn.Module):
    """
    Specialized loss for anatomical region consistency.
    Ensures generated false positives maintain anatomical plausibility.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, generated_features, real_features, mask):
        """
        Calculate anatomical consistency loss between generated and real features
        in non-perturbed regions.
        """
        # Focus on non-masked regions for anatomical consistency
        non_mask = 1 - mask
        
        # Calculate feature consistency in non-perturbed regions
        non_mask_count = non_mask.sum() + 1e-6
        feature_diff = ((generated_features - real_features) * non_mask).abs().sum()
        
        return feature_diff / non_mask_count


class FPGenerationLoss(nn.Module):
    """
    False Positive Generation Loss
    Encourages the generation of high-value false positives that can fool the detector.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, detector_preds, attack_region):
        """
        Calculate loss to maximize false positive generation in attack region.
        detector_preds: predictions from the detector on the generated image
        attack_region: region where we want to generate false positives
        """
        # Extract predictions in the attack region
        x1, y1, x2, y2 = attack_region
        region_preds = []
        
        for pred in detector_preds:
            # Check if prediction overlaps with attack region
            pred_x1, pred_y1, pred_x2, pred_y2 = pred[:4]
            iou = self.calculate_iou([pred_x1, pred_y1, pred_x2, pred_y2], [x1, y1, x2, y2])
            if iou > 0.3:  # Significant overlap
                region_preds.append(pred)
        
        if len(region_preds) == 0:
            # No detections in attack region - maximum loss
            return torch.tensor(1.0, device=self.device)
        
        # Calculate how "polyp-like" the detections are
        confidences = [pred[4] for pred in region_preds]  # Detection confidence
        max_confidence = max(confidences)
        
        # Loss is lower when we have high-confidence detections in attack region
        return 1.0 - max_confidence
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area + 1e-6
        
        return intersection_area / union_area


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py    
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class DADALoss:
    """Enhanced loss function for DADA framework with region-adaptive perturbations and false positive generation focus."""

    sort_obj_iou = False

    def __init__(self, model, autobalance=False, region_adaptive=True, fp_focus_weight=0.2):
        """Initializes enhanced DADA loss with model and region-adaptive perturbation options."""
        device = next(model.parameters()).device  # get model device
        
        # ===== FIXED: Added safe hyperparameter access with default values =====
        h = getattr(model, 'hyp', {})  # Get hyp attribute if exists, otherwise empty dict
        
        # Define criteria with fallback values
        cls_pw = h.get("cls_pw", 1.0)  # Default positive weight for classification
        obj_pw = h.get("obj_pw", 1.0)  # Default positive weight for objectness
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))
        # ===== END FIXED =====

        # Class label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss - with fallback value
        g = h.get("fl_gamma", 0.0)  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # ===== FIXED: Safe access to model components =====
        m = de_parallel(model)
        detect_module = m.model[-1] if hasattr(m, 'model') else m
        
        self.balance = {3: [4.0, 1.0, 0.4]}.get(getattr(detect_module, 'nl', 3), [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(getattr(detect_module, 'stride', [8, 16, 32])).index(16) if autobalance else 0  # stride 16 index
        # ===== END FIXED =====
        
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = getattr(detect_module, 'na', 3)  # number of anchors
        self.nc = getattr(detect_module, 'nc', 80)  # number of classes
        self.nl = getattr(detect_module, 'nl', 3)  # number of layers
        self.anchors = getattr(detect_module, 'anchors', torch.tensor([[[10,13], [16,30], [33,23]]], device=device))
        self.device = device
        
        # DADA-specific enhancements
        self.region_adaptive = region_adaptive
        self.fp_focus_weight = fp_focus_weight
        self.anatomical_loss = AnatomicalRegionLoss(device)
        self.fp_generation_loss = FPGenerationLoss(device)
        self.current_epoch = 0
        self.total_epochs = h.get("epochs", 300)

    def __call__(self, p, targets, features=None, attack_region=None, epoch=0):
        """Performs forward pass with region-adaptive loss calculation."""
        self.current_epoch = epoch
        
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        l_anat = torch.zeros(1, device=self.device)  # anatomical consistency loss
        l_fp = torch.zeros(1, device=self.device)   # false positive generation loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Base detection losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # DADA-specific losses
        if features is not None and attack_region is not None:
            # Extract features for the attack region
            feature_map = features[0]  # Use first feature map
            
            if self.region_adaptive:
                # Classify anatomical region type
                region_type = classify_anatomical_region(feature_map, attack_region)
                
                # Update perturbation factor based on region type
                alpha = get_region_adaptive_alpha(region_type)
                # This alpha would be used outside this class to update the perturbation
                
                # Add anatomical consistency loss
                if hasattr(self, 'real_features'):
                    mask = torch.zeros_like(feature_map)
                    x1, y1, x2, y2 = attack_region
                    mask[:, :, int(y1):int(y2), int(x1):int(x2)] = 1
                    l_anat = self.anatomical_loss(feature_map, self.real_features, mask)
            
            # Add false positive generation loss
            if hasattr(self, 'detector_preds'):
                l_fp = self.fp_generation_loss(self.detector_preds, attack_region)

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        
        # ===== FIXED: Safe hyperparameter access with defaults =====
        # Apply loss weights with epoch-based scheduling
        progress = self.current_epoch / max(self.total_epochs, 1)
        lbox *= self.hyp.get("box", 0.05)
        lobj *= self.hyp.get("obj", 1.0) * (1 + progress * 0.5)  # Increase obj weight as training progresses
        lcls *= self.hyp.get("cls", 0.5)
        l_anat *= 0.1 * (1 - progress * 0.8)  # Decrease anatomical loss weight over time
        l_fp *= self.fp_focus_weight * (0.5 + progress * 0.5)  # Increase FP focus as training progresses
        # ===== END FIXED =====

        bs = tobj.shape[0]  # batch size

        # Total loss with DADA components
        total_loss = (lbox + lobj + lcls + l_anat + l_fp) * bs
        loss_components = torch.cat((lbox, lobj, lcls, l_anat, l_fp)).detach()
        
        # Print adaptive alpha information
        if self.region_adaptive and features is not None and attack_region is not None:
            region_type = classify_anatomical_region(features[0], attack_region)
            alpha = get_region_adaptive_alpha(region_type)
            region_names = ["lumen", "folds", "specular highlights", "flat mucosa"]
            print(f"\rRegion type: {region_names[region_type]}, Adaptive alpha: {alpha:.4f}", end="")

        return total_loss, loss_components

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp.get("anchor_t", 4.0)  # compare with fallback
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch