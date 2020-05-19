import argparse
from sys import platform

#from cellyolo.models import *  # set ONNX_EXPORT in cellyolomodels.py
from cellyolo.utils import *
import models as models
import torchvision.transforms as transforms
from cellyolomodels import *

import torchvision.datasets as datasets
from collections import OrderedDict


labels2classesofSSHD = {
'[0]': 'Healthy',
'[1]': 'Sezary Syndrome'
}

script_dir = os.path.dirname(__file__)

def detect(opt):
    targetlist=[]
    predictionlabel=[]
    pathlist=[]
    TP,FP,FN,TN=0,0,0,0
    img_size = (416, 256)if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    source, weights, =  opt.source, opt.weights,
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    testdir = os.path.join(source, '')
    valdir = os.path.join(source, 'val')
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        checkpoint=torch.load(weights, map_location=device)
        #best_acc1 = checkpoint['best_acc1']
        #start_epoch = checkpoint['epoch']
        modelname=checkpoint['arch']
        print(modelname)
        if modelname not in('ghostresnet'):
            model = models.__dict__[modelname](pretrained=True)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            print('Finished loading model!')
        else:
            model = models.__dict__[modelname]()
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            print('Finished loading model!')

    else:  # darknet format
        breakpoint("weights file wrong formel")

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=10)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Set Dataloader
    vid_path, vid_writer = None, None
    '''
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(valdir, img_size=img_size, half=False)
        print(len(dataset))
    else:
        save_img = True
        dataset = LoadImages(valdir, img_size=img_size, half=False)
        print(len(dataset))
        
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    if os.path.exists(valdir):
        valdataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))
    else:
        valdataset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))

    print(valdataset.imgs)

    for(image, target) in valdataset.imgs:
        imgpath = os.path.join(script_dir, image)
        pathlist.append(imgpath)
    #print(pathlist)



    #print(len(test_loader.dataset))
    '''
    for path, img, im0s, vid_cap in dataset:
        print(path)
        # Get detections
        img = torch.from_numpy(img).to(device)

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        output = model(img)

        #print(len(pred))
        # Process detections
        #prediction = torch.max(pred, 1)[1]
        print("output[0]: ", output[0])

        pred = output.argmax(dim=1, keepdim=True)
        #print("pred[0]: ", pred[0])
        # 得到的prediction为cuda的tensor格式，需要转换为cpu格式，然后将tensor转换为numpy
        #print(labels2classes['1'])
        #print(str(pred[0].cpu().numpy()))
        print(labels2classes[str(pred[0].cpu().numpy())])
        
    '''
    with torch.no_grad():
        end = time.time()
        for i,(images, target) in enumerate(valdataset):
            #print((images, target))
            #print("Label of The Image:",target)

            img = images.unsqueeze(0)
            # compute output
            output = model(img)
            #print(img)
            # print("the images id is:",images)
            #print("Prediction Confidence Score by CellNet: ", output[0][0:2])

            pred = output.argmax(dim=1, keepdim=True)


            if pred[0].cpu().numpy() == 1 and pred[0].cpu().numpy() == target:
                TP=TP+1

            elif pred[0].cpu().numpy() == 0 and pred[0].cpu().numpy() != target:
                FN=FN+1

            elif pred[0].cpu().numpy() == 1 and pred[0].cpu().numpy() != target:
                FP=FP+1

            else:
                TN=TN+1

            targetlist.append('['+str(target)+']')
            predictionlabel.append(str(pred[0].cpu().numpy()))
            #print("Prediction by CellNet: ", labels2classesofSSHD[str(pred[0].cpu().numpy())])

            '''
            if source in("COVID19"):
                targetlist.append(['[' + str(target) + ']'])
                print("Prediction by CellNet: ",labels2classesofCOVID[str(pred[0].cpu().numpy())])
                predictionlabel.append(labels2classesofCOVID[str(pred[0].cpu().numpy())])
                print("targt", labels2classesofSSHD['[' + str(target) + ']'])
                print("prediction:", labels2classesofSSHD[str(pred[0].cpu().numpy())])

            else:
                targetlist.append(labels2classesofSSHD['['+str(target)+']'])
                print("Prediction by CellNet: ", labels2classesofSSHD[str(pred[0].cpu().numpy())])
                predictionlabel.append(labels2classesofSSHD[str(pred[0].cpu().numpy())])
                print("targt",labels2classesofSSHD['['+str(target)+']'])
                print("prediction:",labels2classesofSSHD[str(pred[0].cpu().numpy())])
             '''
        print("TP:", TP)
        print("FN:", FN)
        print("FP:", FP)
        print("TN:", TN)
        print("Precision:", (TP / (TP + FP)))
        print("Recall:", (TP / (TP + FN)))

        #print("Accuracy on Val:", (TP + TN) / (TP + TN + TP + FN))
        print("Specificity:", (TN / (TN + FP)))
        print("Accuracy :", (TP+TN)/(TP + TN + TP + FN))
        print("the lengths of targetlist:",len(targetlist))
        print("the lengths of predictionlabel:", len(predictionlabel))
        print("the lengths of path:", len(pathlist))


    return predictionlabel,targetlist,(TP / (TP + FP)),(TP / (TP + FN)),pathlist

    '''
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            print(test_loader.dataset.imgs)
            print("i:", i)


            # compute output
            output = model(images)
            # print("the images id is:",images)
            print("output[0]: ", output[0][0:2])
            pred = output.argmax(dim=1, keepdim=True)
            print(labels2classes[str(pred[0].cpu().numpy())])
    
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='cellyolo/weights/model_best_weights.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='HDSS/', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
