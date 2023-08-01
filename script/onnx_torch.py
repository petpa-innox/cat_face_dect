import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
 
onnx_model_path = "/home/holmes/code/cat_face_detection/test.onnx"
 
# https://pytorch.org/hub/pytorch_vision_densenet/
model = torch.hub.load('pytorch/vision', 'resnet50', weights='ResNet50_Weights.DEFAULT')
 
# set the model to inference mode
model.eval()
 
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
 model,
 dummy_input,
 onnx_model_path,
 verbose=True,
 input_names=["input"],
 output_names=["output"],
 opset_version=11
)