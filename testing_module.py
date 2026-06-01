from datasetLoader import BDDDataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from valuation import Evaluator
from transforms import Compose, Resize, ToTensor


def collate_fn(batch):
        return tuple(zip(*batch))

def main(dataset):
    num_classes = 11
    transforms = Compose([Resize((384, 640)),ToTensor()])
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    model.load_state_dict(torch.load(f"best_model/best_model_{dataset}.pth"))

    device = torch.device("cuda")
    model.to(device)

    train_dataset = BDDDataset(
        root=f"{dataset}/images/train",
        annotation_file=f"{dataset}/labels/train.json",
        transforms=transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_dataset = BDDDataset(
        root=f"{dataset}/images/val",
        annotation_file=f"{dataset}/labels/val.json",
        transforms=transforms
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    night_dataset = BDDDataset(
        root=f"{dataset}/images/night_val",
        annotation_file=f"{dataset}/labels/night_val.json",
        transforms=transforms
    )

    night_loader = DataLoader(
        night_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    evaluator=Evaluator()

    print("Evaluating TRAIN set...")

    train_results = evaluator.evaluate(
        model,
        train_loader,
        device
    )
    for k,v in train_results.items():
        print(k,":",v)

    print("Evaluating VAL set...")

    val_results = evaluator.evaluate(
        model,
        val_loader,
        device
    )

    for k,v in val_results.items():
        print(k,":",v)

    print("Evaluating NIGHT set...")

    night_results = evaluator.evaluate(
        model,
        night_loader,
        device
    )

    for k,v in night_results.items():
        print(k,":",v)

if __name__ == "__main__":
    main(dataset="mini_dataset")