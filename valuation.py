import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Evaluator():
    
    @torch.no_grad()
    def evaluate(self, model, loader, device):

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

            metric.update(preds, gts)

        return metric.compute()