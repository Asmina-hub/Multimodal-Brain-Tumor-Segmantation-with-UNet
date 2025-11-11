import segmentation_models_3D as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from image_segmentation_with_u_net import UNet
from  torch.utils.data import  DataLoader
from dataloader import CustomDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from monai.losses import DiceLoss, FocalLoss
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm

def get_losses_and_metrics(num_classes=4):
    dice_fn = DiceLoss(to_onehot_y=True, softmax=True)
    focal_fn = FocalLoss(to_onehot_y=True, gamma=2.0)

    def total_loss(outputs, masks):
        return dice_fn(outputs, masks) + focal_fn(outputs, masks)
    
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")

    return total_loss, iou_metric, acc_metric

#-----------------------------------#

train_img_dir = "dataset/BraTS2020_TrainingData/input_data_3channels/images_train"
train_mask_dir = "dataset/BraTS2020_TrainingData/input_data_3channels/masks_train"
val_img_dir = "dataset/BraTS2020_TrainingData/input_data_3channels/images_val"
val_mask_dir = "dataset/BraTS2020_TrainingData/input_data_3channels/mask_val"
train_dataset = CustomDataset(train_img_dir, train_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print("Training data loader created with {} samples.".format(len(train_dataset)))
val_dataset = CustomDataset(val_img_dir, val_mask_dir)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
print("Test data loader created with {} samples.".format(len(val_dataset)))

criterion, iou_metric, acc_metric = get_losses_and_metrics(num_classes=4)
model = UNet(in_channel=3, class_no=4)
LR =0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

##-----------------------------------#

def training(model, train_loader, val_loader, optimizer, criterion, iou_metric, acc_metric, num_epochs, device=None, save_dir="training_plots"):
    """
    Modular training loop that works with external loss and metric functions.
    """

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_ious, val_ious = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        model.train()
        running_train_loss = 0.0

        # reset metrics
        acc_metric.to(device).reset()
        iou_metric.to(device).reset()

        # ------------------- TRAINING -------------------
        for images, masks in tqdm(train_loader, desc="Training", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            labels = masks.long()
            acc_metric.update(preds, labels)
            iou_metric.update(preds, labels)

        # compute train metrics
        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = acc_metric.compute().item()
        train_iou = iou_metric.compute().item()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_ious.append(train_iou)

        # ------------------- VALIDATION -------------------
        model.eval()
        running_val_loss = 0.0
        acc_metric.to(device).reset()
        iou_metric.to(device).reset()

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                labels = masks.long()
                acc_metric.update(preds, labels)
                iou_metric.update(preds, labels)

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = acc_metric.compute().item()
        val_iou = iou_metric.compute().item()

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_ious.append(val_iou)

        # ------------------- LOGGING -------------------
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}"
        )

    # ------------------- SAVE METRICS -------------------
    np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(save_dir, "train_accs.npy"), np.array(train_accs))
    np.save(os.path.join(save_dir, "val_accs.npy"), np.array(val_accs))
    np.save(os.path.join(save_dir, "train_ious.npy"), np.array(train_ious))
    np.save(os.path.join(save_dir, "val_ious.npy"), np.array(val_ious))
    # plot and save figures
    epochs = np.arange(1, num_epochs+1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="train_acc")
    plt.plot(epochs, val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_ious, label="train_iou")
    plt.plot(epochs, val_ious, label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "iou_curve.png"))
    plt.close()

    # return histories for further use
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_ious": train_ious,
        "val_ious": val_ious
    }


num_epochs = 25
training_history = training(
    model, train_loader, val_loader, optimizer, criterion, iou_metric, acc_metric,
    num_epochs=num_epochs, device=None, save_dir="training_plots"
)   

