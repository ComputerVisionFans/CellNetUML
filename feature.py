import torch
from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import json
import models as models

model = models.Cell_net_8layer()
# print(model)
model.eval()
image = Image.open('covid.jpg')
# transformation
#tf = transforms.Compose([transforms.Resize((224, 224)),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                             std=[0.229, 0.224, 0.225])])

tf= transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()])
img = tf(image)
img = img.unsqueeze(0)


def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

#for resnet
#f1 = model.conv1(img)  # [1, 64, 112, 112]
#save_img(f1,'conv1')

'''
#for resnet
print(model.squeeze[0])
#f1 = model.features[0][0:2](img)  # [1, 64, 224, 224]
#f1 = model.features[0:11](img)  # [1, 64, 122, 122] contain 10 layers
f1=model.classifier[0](model.squeeze(model.features(img)))
save_img(f1,'Conv2d+avgpool+Conv2d')
'''
#print(model.features)
#new_model = model.features[:15]
#print(new_model)
#f7 = new_model(img)  # [1, 256, 14, 14]
#save_img(f7, '9th conv2d+BN')

new_model = nn.Sequential(*list(model.children())[:2])
print(new_model)
f3 = new_model(img)
#print(new_model)
#save_img(f3, '9th 1X1conv2d+BN+Relu')
'''
#提取出模型的前几层
new_model = nn.Sequential(*list(model.children())[:4])
f2 = new_model(img)  # [1, 64, 56, 56]
f2 = model.layer1[0](f2)
layer1_conv1 = model.layer1[1].conv1
layer1_bn = model.layer1[1].bn1
layer1_relu = model.layer1[1].relu
layer1_conv2 = model.layer1[1].conv2

f2 = layer1_conv1(f2)
save_img(f2, 'layer1_conv3')
f2 = layer1_conv2(layer1_relu(layer1_bn(f2)))
save_img(f2, 'layer1_conv4')

new_model = nn.Sequential(*list(model.children())[:5])
f3 = new_model(img)
save_img(f3, 'layer1')
#
new_model = nn.Sequential(*list(model.children())[:6])
f4 = new_model(img)  # [1, 128, 28, 28]
save_img(f4, 'layer2')
#
new_model = nn.Sequential(*list(model.children())[:7])
print(new_model)
f5 = new_model(img)  # [1, 256, 14, 14]
print(f5.shape)
save_img(f5, 'layer3')
#
new_model = nn.Sequential(*list(model.children())[:8])
print(new_model)
f6 = new_model(img)  # [1, 256, 14, 14]
print(f6.shape)
save_img(f6, 'layer4')
#
new_model = nn.Sequential(*list(model.children())[:9])
print(new_model)
f7 = new_model(img)  # [1, 256, 14, 14]
save_img(f7, 'avgpool')
'''