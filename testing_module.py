from datasetLoader import BDDDataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from valuation import Evaluator
from transforms import Compose, Resize, ToTensor

def collate_fn(batch):
        return tuple(zip(*batch))

def main(dataset, lr, epochs):
    num_classes = 11
    transforms = Compose([Resize((384, 640)),ToTensor()])
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    model.load_state_dict(torch.load(f"best_model/best_model_{dataset}_{epochs}_epochs_{str(lr)[2:]}_lr.pth"))

    device = torch.device("cuda")
    model.to(device)

    def val_dataset_eval(split="val", evaluator=None):
        val_dataset = BDDDataset(
            root=f"{dataset}/images/{split}",
            annotation_file=f"{dataset}/labels/{split}.json",
            transforms=transforms
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        if evaluator is None:
            evaluator=Evaluator()
        print(f"Evaluating {split.upper()} set...")
        val_results,pre_recall,class_report = evaluator.evaluate(
            model,
            val_loader,
            device
        )
        for k,v in val_results.items():
            print(k,":",v)
        print("Precision:", pre_recall["precision"])
        print("Recall:", pre_recall["recall"])
        print("Classification Report:")
        print(class_report)

    val_dataset_eval(split="val",evaluator=Evaluator())
    val_dataset_eval(split="test_day",evaluator=Evaluator())
    val_dataset_eval(split="test_night",evaluator=Evaluator())

if __name__ == "__main__":
    main(dataset="main_dataset_2k",lr=0.0001,epochs=30)