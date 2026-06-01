import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from datasetLoader import BDDDataset
from transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def main(dataset):
    lr=0.01
    batch_size=4
    num_workers=4
    train_transforms = Compose([Resize((384, 640)),ToTensor()])
    train_dataset=BDDDataset(root=f"{dataset}/images/train",annotation_file=f"{dataset}/labels/train.json",transforms=train_transforms)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,persistent_workers=True,collate_fn=collate_fn)
    print("Loaded Dataset...")
    num_classes = 11   # 10 classes + background

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    # model.load_state_dict(torch.load(f"best_model/best_model_{dataset}.pth"))

    device = torch.device("cuda")
    model.to(device)
    print("Initialized model...")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    scaler = torch.amp.GradScaler("cuda")
    epochs=30
    model.train()
    print("Started training...")
    best_loss=float("inf")
    loss_vs_epoch=[]
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} training...")
            epoch_loss=0
            i=1
            for images, targets in train_loader:
                if i%100==0: 
                    print(f"Trained on {i} batches(size=4)")
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()}
                    for t in targets
                ]
                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):

                    loss_dict = model(
                        images,
                        targets
                    )

                    losses = sum(
                        loss
                        for loss in loss_dict.values()
                    )

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss+=losses.item()
                i+=1
            if epoch_loss<best_loss:
                best_loss=epoch_loss
                torch.save(model.state_dict(), f"best_model/best_model_{dataset}.pth")
            print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader)}")
            loss_vs_epoch.append(epoch_loss)
    except KeyboardInterrupt:
        print("Stopping Training...")
        print("="*60)
    plt.plot(list(range(1,epochs+1)),loss_vs_epoch)
    plt.title(f"Loss vs Epoch (dataset:{dataset})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"results/losses_{dataset}.png")
    print(f"best model saved at: best_model/best_model_{dataset}.pth")
    print(f"Loss plot saved at: results/losses_{dataset}.png")

if __name__=='__main__':
    main(dataset="mini_dataset")