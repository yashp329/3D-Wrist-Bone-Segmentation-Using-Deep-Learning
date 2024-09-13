# IMPORTING LIBRARIES
import os
from glob import glob
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityRanged, RandFlipd,
    RandAffined, RandRotated, RandZoomd, NormalizeIntensityd, ToTensord, EnsureTyped, EnsureType
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism, first
from monai.networks.nets import DynUNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Activations
import torch
import numpy as np
from monai.transforms import MapTransform

print("All libraries have been successfully Imported!!!")

#=========================================================================
## CREATING PATHS TO UPLOAD OUR DATA
data_dir = '/cs/home/psxyp14/wrist_bone_Unetr/data'

train_images = sorted(glob(os.path.join(data_dir,'imagesTr','*nii.gz')))
train_labels = sorted(glob(os.path.join(data_dir,'labelsTr','*nii.gz')))

val_images = sorted(glob(os.path.join(data_dir,'imagesTs','*nii.gz')))
val_labels = sorted(glob(os.path.join(data_dir,'labelsTs','*nii.gz')))

train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in  zip(train_images,train_labels)]
val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in  zip(val_images,val_labels)]


orig_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    ToTensord(keys=['image', 'label'])
])

# Defining training transforms
train_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
    ScaleIntensityRanged(keys=['image'], a_min=0.0011, a_max=1.01, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.5),
    RandAffined(keys=['image', 'label'], prob=0.5, rotate_range=(0, 0, np.pi/12), scale_range=(0.1, 0.1, 0.1),
                mode=("nearest", "nearest")),  # Apply nearest neighbors interpolation for both images and labels
    RandRotated(keys=['image', 'label'], prob=0.5, range_x=np.pi/12, range_y=np.pi/12, range_z=np.pi/12, mode="nearest"),
    RandZoomd(keys=['image', 'label'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode="nearest"),
    EnsureTyped(keys=['image', 'label']),  # Ensure the data is of the correct type
    # CastToIntd(keys=['label']),  # Cast the label data to integers
    ToTensord(keys=['image', 'label'])
])

# Defining validation transforms
val_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
    ScaleIntensityRanged(keys=['image'], a_min=0.0011, a_max=1.01, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=['image', 'label']),  
    ToTensord(keys=['image', 'label'])
])


# Creating datasets and dataloaders
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)
orig_ds = Dataset(data=train_files, transform=orig_transforms)

orig_ds = Dataset(data=train_files, transform=orig_transforms)
orig_loader = DataLoader(orig_ds, batch_size =2)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size =2)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size =2)

print("Data Loaded Successfully!!!")

#==========================================================================

test_patient = first(train_loader)
orig_patient = first(orig_loader)

print(torch.min(test_patient['image']))
print(torch.max(test_patient['image']))


sample_label = test_patient['label'][0, 0, :, :, 15].cpu().numpy()
unique_values = np.unique(sample_label)
print(f"Unique label values: {unique_values}")


# Checking if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the Dynamic Unet model
model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=11,  # 10 bones + background
    kernel_size=[3, 3, 3, 3],  # Using 4 elements to match the number of strides
    strides=[1, 2, 2, 2],  # Adjusted strides to avoid negative padding
    upsample_kernel_size=[2, 2, 2],  # Changed to list
    norm_name="instance",
    deep_supervision=False,
    res_block=True,
).to(device)

# Defining the loss function
loss_function = DiceLoss(to_onehot_y=True, softmax=True)

# Defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Defining metrics
dice_metric = DiceMetric(include_background=True, reduction="mean")
post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=11)])
post_label = Compose([AsDiscrete(to_onehot=11)])

print("Model, loss function, optimizer, and metrics are set up.")


#====================================================================
print("Model Training Successfully started!!!")

# Early stopping parameters
patience = 50  # Number of epochs with no improvement after which training will be stopped
early_stop = False
counter = 0  # Counter for early stopping

# Training loop
max_epochs = 300 
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# Initializing dictionary to store the best metrics for each class
best_class_metrics = {i: 0 for i in range(11)}  # assuming 11 classes including background

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_loader)} train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_images, (128, 128, 48), 4, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)
            
            # Calculating metrics
            metric = dice_metric.aggregate().item()
            metric_batch = dice_metric_batch.aggregate()
            dice_metric.reset()
            dice_metric_batch.reset()
            
            metric_values.append(metric)

            # Saving the best model based on the mean Dice metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_dyunet.pth")
                print("saved new best metric model")
                counter = 0  # Reset early stopping counter
                # Updating best metrics for each class
                for i in range(11):
                    best_class_metrics[i] = metric_batch[i].item()
            else:
                counter += 1  # Increment early stopping counter
            
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")
            for i, dice in enumerate(metric_batch):
                print(f"Class {i} dice: {dice.item():.4f}")
            print(f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
            
            # Early stopping check
            if counter >= patience:
                print("Early stopping triggered")
                early_stop = True
                break

    if early_stop:
        break

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# Loading the best model for final evaluation
model.load_state_dict(torch.load("best_metric_model_dyunet.pth"))
model.eval()

# Evaluating and printing the dice coefficient for every label achieved through the best model
with torch.no_grad():
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    for val_data in val_loader:
        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        val_outputs = sliding_window_inference(val_images, (128, 128, 48), 4, model)
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        dice_metric_batch(y_pred=val_outputs, y=val_labels)
    
    metric_batch = dice_metric_batch.aggregate()
    dice_metric_batch.reset()

    print("Final per-class Dice metrics:")
    for i, dice in enumerate(metric_batch):
        print(f"Class {i} dice: {dice.item():.4f}")
