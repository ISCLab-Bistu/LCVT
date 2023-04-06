# LCVT
Lightweight Convolutional Vision Transformer Network for Detecting Outfalls into Rives and Oceans in Aerial Images


## Introduction


This repository contains the official implementation of **Lightweight Convolutional Vision Transformer Network for Detecting Outfalls into Rives and Oceans in Aerial Images**. The purpose of our work is to propose a lightweight model that is specifically designed for detecting outfalls in UAV aerial images, which can be easily deployed on software for detecting outfalls. This model efficiently assists experts in quickly identifying and locating areas of outfalls in UAV aerial images. This work was completed by Intelligent Sensing and Computing Laboratory affiliated to Beijing Information Science and Technology University. It has been submitted to the Journal of Neurocomputing published by Elsevier.

<p align="center">
9
  <img src="https://github.com/ISCLab-Bistu/LCVT/blob/main/image/vis2.jpg" />
10
</p>
11
​
12
<p align = "center">
13
Adaptive Spatial Correlation Pyramid Attention (ASCPA). 
14
</p>
15
​
16
The structure of the ASCPA network is shown in above picture. This network is mainly composed of SPE (Spatial Pyramid Extractor) and SCFM (Spatial Correlation Fusion Module). The purpose of the SPE is to extract multi-scale spatial information on the feature map. The SCFM is used to perform spatial correlation feature recalibration to selectively emphasized informative features. 
17
​
18
​
19
## Result
20
​
21
<p align="center">
22
    <img src="https://github.com/ISCLab-Bistu/LCVT/tree/main/image/vis1.jpg" />
23
</p>
24
​
25
<p align = "center">
26
Visualization results are shown in above picture using Gradcam tool. Obviously, the heat map of YOLOv5-ASCPA is completely able to cover the core area of the outfall which reflects higher response intensity with dark red. The error-prone region has a very weak response intensity for the YOLOv5-ASCPA model compared with the other three models.
27
</p>
28
​
29
## Usage
30
​
31
ASCPA can be inserted into [YOLOv3](https://github.com/ultralytics/yolov3), [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) and [YOLOv5](https://github.com/ultralytics/yolov5/) as a standalone module.
32
​
33
The [configuration file](config/yolov5s.yaml) for YOLOv5 is provided here as an example.
34
​
35
## Summary & Prospect
36
​
37
Our team proposed ASCPA in order to promote the capability of YOLO series network for small object detection. ASCPA was tested on our outfall dataset and proved its excellent capability. Despite that ASCPA has not been tested in other datasets, we still hope that it will demonstrate his power in more datasets.
