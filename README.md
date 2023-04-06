# LCVT
Lightweight Convolutional Vision Transformer Network for Detecting Outfalls into Rives and Oceans in Aerial Images


## Introduction


This repository contains the official implementation of **Lightweight Convolutional Vision Transformer Network for Detecting Outfalls into Rives and Oceans in Aerial Images**. The purpose of our work is to propose a lightweight model that is specifically designed for detecting outfalls in UAV aerial images, which can be easily deployed on software for detecting outfalls. This model efficiently assists experts in quickly identifying and locating areas of outfalls in UAV aerial images. This work was completed by Intelligent Sensing and Computing Laboratory affiliated to Beijing Information Science and Technology University. It has been submitted to the Journal of Neurocomputing published by Elsevier.

<p align="center">
    <img src="https://github.com/ISCLab-Bistu/LCVT/blob/main/image/vis2.jpg" />
</p>



<p align = "center">

Adaptive Spatial Correlation Pyramid Attention (ASCPA). 

</p>



The structure of the ASCPA network is shown in above picture. This network is mainly composed of SPE (Spatial Pyramid Extractor) and SCFM (Spatial Correlation Fusion Module). The purpose of the SPE is to extract multi-scale spatial information on the feature map. The SCFM is used to perform spatial correlation feature recalibration to selectively emphasized informative features. 







## Result

# Visulization 1


<p align="center">
    <img src="https://github.com/ISCLab-Bistu/LCVT/blob/main/image/vis1.jpg" />
</p>

<p align = "center">

Visualization results are shown in above picture using Gradcam tool. Obviously, the heat map of Our proposed model is completely able to cover the core area of the outfall which reflects higher response intensity with dark red. The error-prone region has a very weak response intensity for our proposed model compared with the other models.

</p>


# Visulization 2


<p align="center">
    <img src="https://github.com/ISCLab-Bistu/LCVT/blob/main/image/vis2.jpg" />
</p>

<p align = "center">

Obviously, the heat map of the model with LCVT is completely able to cover the core area of the outfall which reflects higher response intensity with dark red. The error-prone region has a very weak response intensity for the model with compared with the other models.

</p>

## Usage



ASCPA can be inserted into [YOLOv3](https://github.com/ultralytics/yolov3), [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) and [YOLOv5](https://github.com/ultralytics/yolov5/) as a standalone module.



The [configuration file](config/yolov5s.yaml) for YOLOv5 is provided here as an example.



## Summary & Prospect



Our team proposed ASCPA in order to promote the capability of YOLO series network for small object detection. ASCPA was tested on our outfall dataset and proved its excellent capability. Despite that ASCPA has not been tested in other datasets, we still hope that it will demonstrate his power in more datasets.
