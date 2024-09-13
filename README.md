# 3D-Wrist-Bone-Segmentation-Using-Deep-Learning

<div align="center">
  <img src="https://github.com/yashp329/3D-Wrist-Bone-Segmentation-Using-Deep-Learning/blob/main/Dissertation_Results.png" alt="Wrist Bone Segmentation">
</div>

## Project Overview

Medical image segmentation plays a critical role in improving diagnosis and treatment, especially in orthopaedics. This project focuses on automating the segmentation of 10 wrist bones from 3D MRI scans using advanced deep learning techniques. By developing this automated segmentation system, clinicians can speed up their workflows, reduce manual errors, and improve the accuracy of diagnosis for conditions like fractures, arthritis, and other wrist-related issues.

## Key Features
- **Advanced Deep Learning Models**: Implemented state-of-the-art models such as nnUNet, DynUNet, SegResNet, UNETR, and SwinUNETR for precise 3D segmentation.
- **Ensemble Approach**: Combined the strengths of UNETR, SegResNet, and DynUNet to create a more accurate and robust segmentation system.
- **Frameworks**: Leveraged MONAI for efficient data preprocessing and PyTorch for scalable model development.
- **Metrics for Performance**: Evaluated model performance using industry-standard metrics like Dice Score and ASSD, ensuring high-quality results.

## Dataset
- **Data**: The dataset consists of 50 expert-annotated 3D MRI scans of the wrist, with each scan having a resolution of 128x128x48.
- **Annotations**: 10 wrist bones, including the scaphoid, lunate, and triquetrum, are accurately labeled for training and testing.

## Preprocessing
- **MONAI Framework**: Efficiently handled 3D data preprocessing with operations such as intensity normalization, scaling, and random affine transformations.
- **Data Augmentation**: Applied random rotations, scaling, and affine transformations to improve model generalization and performance.
  
## Results
Our models achieved excellent performance, as evaluated by the Dice Score and ASSD metrics:

- **nnUNet**: Achieved a Dice score of 0.8969, demonstrating precise segmentation across all wrist bones.
- **Ensemble (UNETR, SegResNet, DynUNet)**: Combined the strengths of multiple models, attaining a Dice score of 0.8966, enhancing robustness in handling complex anatomical variations.

### Example Segmentations:
1. **nnUNet** segmentation of the Ulna (Dice Score: 0.8724):  
2. **Ensemble** segmentation of the Scaphoid (Dice Score: 0.8934):  

## Conclusion
By employing multiple state-of-the-art deep learning models and an ensemble method, this project successfully demonstrated accurate segmentation of wrist bones from 3D MRI scans. The results show potential for integration into clinical workflows, aiding in faster and more accurate diagnoses. Both nnUNet and the ensemble method offer significant improvements in handling complex wrist anatomies, with the ability to apply these models to other orthopaedic diagnostic tools in the future.
