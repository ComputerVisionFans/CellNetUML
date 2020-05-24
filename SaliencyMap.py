
#IMPORTS

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import models as models
import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image
import glob, os
#Using VGG-19 pretrained model for image classification
imagefolder=  'saliencymap/'

model = models.ghostresnet()
checkpoint=torch.load('cellyolo/weights/checkpoint2.pt',map_location=torch.device('cpu'))
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})

print('Finished loading model!')
for param in model.parameters():
    param.requires_grad = False


def download(url, fname):
    response = requests.get(url)
    with open(fname, "wb") as f:
        f.write(response.content)


# Downloading the image
#download("C://Users//qianli//PycharmProjects//CellNetUML//COVID19//val//NORMAL//", "IM-0001-0001.jpeg")

# Opening the image
#img = Image.open('C://Users//qianli//PycharmProjects//CellNetUML//COVID19//val//NORMAL//IM-0001-0001.jpeg')

# Downloading the image
##download("https://specials-images.forbesimg.com/imageserve/5db4c7b464b49a0007e9dfac/960x0.jpg?fit=scale","input.jpg")

# Opening the image
#img = Image.open('input.jpg')
#print(img)
#img3 = Image.open('C://Users//qianli//PycharmProjects//CellNetUML//cellyolo//output//hd3 (9).png').convert('RGB')
#img2 = Image.open('C://Users//qianli//PycharmProjects//CellNetUML//HDSS//val//ss//ss1_1 (3).png')
#print(img2)
# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''
def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))

def search(path=".", name="1"):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search(item_path, name)
        elif os.path.isfile(item_path):
            if name in item:
                print(item_path)
                return item_path


def saliencymap():
    for pathAndFilename in glob.iglob(os.path.join(imagefolder, "*.png")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        print(title)
        #print(search(r'C:\Users\qianli\Downloads\cellyolov3newest\content\yolov3',title+".png"))

        img = Image.open(pathAndFilename).convert('RGB')
        # preprocess the image
        X = preprocess(img)
        # we would run the model in evaluation mode
        model.eval()
        # we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
        X.requires_grad_()
        # forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end and we also don't need softmax, we need scores, so that's perfect for us.
        scores = model(X)
        # Get the index corresponding to the maximum score and the maximum score itself.
        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]
        # backward function on score_max performs the backward pass in the computation graph and calculates the gradient of
        # score_max with respect to nodes in the computation graph
        score_max.backward()
        # Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
        # R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
        # across all colour channels.
        saliency, _ = torch.max(X.grad.data.abs(), dim=1)
        # code to plot the saliency map as a heatmap
        plt.imshow(saliency[0], cmap=plt.cm.hot)
        plt.axis('off')
        plt.savefig(str('saliencymap/'+title+"ournetWithoutcellyolo.jpg"))





saliencymap()
