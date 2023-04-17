import argparse
import torch
import os

# from utils.datasets import *
# from utils.utils import *


def detect(save_img=False):
    weights = 'last_2×2_paiwuba_3.pt'
    # Initialize
    device = 'cpu'

    model = torch.load(weights, map_location=device)
    model = model['model']
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse() Focus model.0.conv model.22 Detect
    model.to(device).eval()
    # print(model)
    # for name in model.state_dict():
    #     print(name)
    # Half precision

    img = torch.zeros((1, 3, 640, 640), device=device)  # init img

    torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                      opset_version=11,
                      input_names = ['Focus'],   # the model's input names
                      output_names = ['Detect']
                      )
    

if __name__ == '__main__':

    detect()
