# 3D-Wrist-Bone-Segmentation-Using-Deep-Learning

## Project Overview
This project automates the segmentation of 10 wrist bones from 3D MRI scans, improving the diagnosis of conditions like fractures and arthritis. We employed deep learning models to enhance accuracy and efficiency in the segmentation process.

## Key Features
- **Models**: nnUNet, DynUNet, SegResNet, UNETR, SwinUNETR.
- **Ensemble**: Combined UNETR, SegResNet, and DynUNet for optimal results.
- **Frameworks**: Implemented with MONAI for data preprocessing and PyTorch for model training.
- **Metrics**: Performance measured using Dice Score and ASSD, achieving a Dice score of 0.90.

## Dataset
- 3D MRI scans (128x128x48) with expert-annotated wrist bone segmentations.

## Results
- **nnUNet**: Dice score of 0.8969 for accurate segmentation across all wrist bones.
- **Ensemble**: Achieved similar performance (Dice score of 0.8966), combining the strengths of UNETR, SegResNet, and DynUNet for improved segmentation.

## Conclusion
Both nnUNet and the ensemble method demonstrated excellent segmentation results. These methods can significantly aid in orthopaedic diagnostics by providing accurate and efficient wrist bone segmentations from 3D MRI scans.
