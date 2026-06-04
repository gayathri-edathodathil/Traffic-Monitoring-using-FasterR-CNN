import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import numpy as np
from sklearn.metrics import classification_report
class Evaluator():
    
    @torch.no_grad()
    def evaluate(self, model, loader, device):
        TP = 0
        FP = 0
        FN = 0

        iou_threshold = 0.5
        all_true = []
        all_pred = []

        model.eval()

        metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

        for images, targets in loader:

            images = [img.to(device) for img in images]
            gts = [{k: v.cpu() for k, v in t.items()} for t in targets]
            predictions = model(images)

            preds = []
            for p in predictions:
                pred = {
                    "boxes": p["boxes"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                    "labels": p["labels"].detach().cpu()
                }
                preds.append(pred)

            for pred, target in zip(preds, gts):

                pred_boxes = pred['boxes']

                keep = pred["scores"] >= iou_threshold

                pred_boxes = pred["boxes"][keep]
                pred_labels = pred["labels"][keep]
                pred_scores = pred["scores"][keep]

                true_boxes = target['boxes']

                if len(pred_boxes) == 0:
                    FN += len(true_boxes)
                    continue

                ious = box_iou(pred_boxes, true_boxes)

                matched_gt = set()

                for i in range(len(pred_boxes)):

                    max_iou, gt_idx = ious[i].max(0)

                    if max_iou >= iou_threshold:
                        if gt_idx.item() not in matched_gt:
                            all_true.append(target['labels'][gt_idx].item())
                            all_pred.append(pred_labels[i].item())
                            if pred_labels[i] == target['labels'][gt_idx]:
                                TP += 1
                                matched_gt.add(gt_idx.item())
                        

                    else:
                        FP += 1

                FN += len(true_boxes) - len(matched_gt)
            metric.update(preds, gts)

        precision = np.round(TP / (TP + FP + 1e-6), 4)
        recall = np.round(TP / (TP + FN + 1e-6), 4)
        
        return (metric.compute(), {"precision": precision, "recall": recall}, classification_report(all_true, all_pred, digits=4))