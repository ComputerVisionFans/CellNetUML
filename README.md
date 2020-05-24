# CellNet and CellNet Software
This is the official implementation with *training* code for Cell Morphology Based Diagnosis of Cancer usingConvolutional Neural Networks: CellNet. For technical details, please refer to:

**Cell Morphology Based Diagnosis of Cancer usingConvolutional Neural Networks: CellNet** <br />
[Qiang Li](https://www.linkedin.com/in/qiang-li-166362143/)\*, [Otesteanu Corin](https://biol.ethz.ch/en/the-department/people/person-detail.MTkyNzk5.TGlzdC80NjAsOTIzMDMxMjIy.html)\* <br />
**Paper In preparation for ICPR2020** <br />
**[[Software Report](https://drive.google.com/file/d/1_lVADVGAhkRG2qvzbhACQ5JqrCPpYRwM/view?usp=sharing)] [[CellNetSoftware Video](https://drive.google.com/open?id=1V-VDtiOJv1WI5jVSQZqpj-6AKPFUBexX)] [[Research Grant Page](https://ethz.ch/en/studies/non-degree-courses/exchange-and-visiting-studies/programmes/exchange-programmes/idea-league.html)]** <br />

## Results
These are the reproduction results from this repository. All the training/testing lsg log file on ETH Zurich leonhard cluster can be downloaded from our [github](https://github.com/Johnny-liqiang/CellNetUML/tree/master/training%20log%20file%20for%20verification).

### Evaluate performance on [CiFar10](https://www.cs.toronto.edu/~kriz/cifar.html)
This is the Boxplot of resnet18, Ournet, ghostnet on cifar without Cellyolo. Due to the fact that every image from this dataset is 32*32 pixel image, it's getting hard to train a well segmentor by cellyolo to filter out the other artifacts in the image. As it illustrated that, even without Cellyolo preprocessing, our net already achieved the best performance. 

|<sub>Model</sub>|<sub>Weights(million)</sub>|<sub>Top-1 Val Acc.(%)</sub>|<sub>FLops(million)</sub>|   
| :---: | :---: | :---: | :---: |
| <sub>[VGG-16](https://arxiv.org/abs/1409.1556)</sub> | <sub>15</sub> | <sub>93.6</sub> | <sub>313</sub>|        
| <sub>[ResNet-18](https://arxiv.org/abs/1512.03385)</sub> | <sub>11</sub> | <sub>88.779</sub> | <sub>180</sub> | 
| <sub>[GhostNet](https://arxiv.org/abs/1911.11907)</sub> | <sub>5.18</sub> | <sub>88.238</sub> | <sub>141</sub> | 
| <sub>[OurNet](https://github.com/Johnny-liqiang/CellNetUML)</sub> | <sub>2.91</sub> | <sub>92.45</sub> | <sub>41.7</sub> |


CIFAR-10  dataset  consists of 60,000 32 × 32 color images in 10 classes, with 50,000 training images and 10,000 test images.  A common  data  augmentation  scheme  including random crop and mirroring is adopted as well.

**Note:** 
- Speed are tested on a ETH Zurich Leonhard Cluster. 

<center>
    <img src="https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/Boxplot%20of%20resnet18%20%2C%20OurGhostRe%2C%20ghostnet%20on%20cifar%20without%20cellyolo%20-%20Summary%20Report.png">
</center>

### Evaluate performance on [Pneumonia Dataset](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site)

On  benchmark  pneumonia  dataset,  the  Pneumonia/Normal classification val accuracy of our Net converges into nearly 91.785% better than Ghost Net and ResNet18, In addition, after around 80 epochs the accuracy of our Net converged, comparing to  [Inception  V3](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site)  after 7000 epochs reaches 88.0%.

|<sub>Model</sub>|<sub>Weights(million)</sub>|<sub>Top-1 Val Acc.(%)</sub>|<sub>FLops(million)</sub>|   
| :---: | :---: | :---: | :---: |
| <sub>[InceptionV3](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site)</sub> | <sub>23.81</sub> | <sub>88</sub> | <sub>540</sub>|  
| <sub>[ResNet-18](https://arxiv.org/abs/1512.03385)</sub> | <sub>11</sub> | <sub>87.50</sub> | <sub>180</sub> | 
| <sub>[GhostNet](https://arxiv.org/abs/1911.11907)</sub> | <sub>5.18</sub> | <sub>88.69</sub> | <sub>141</sub> | 
| <sub>[OurNet](https://github.com/Johnny-liqiang/CellNetUML)</sub> | <sub>2.91</sub> | <sub>91.78</sub> | <sub>41.7</sub> |
 

![描述](https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/Time%20Series%20Plot%20of%20ghost%20net%20on%2C%20resnet18%20on%20%2C%20our%20net%20on%20pneumonia%20dataset%20without%20cellyolo.png)

### Evaluate performance on [Sezary Syndrome Dataset](https://github.com/Johnny-liqiang/CellNetUML/tree/master/HDSS)

ResNet18  [17]  and  shuffleNetv2  [25]  were  verified  so  farthe most representative best performance on Sezary SyndromeDataset. But Our* Net can achieve higher classification perfor-mance  (e.g.  95.638%  top-1  accuracy  )  than  ResNet  18  [17],ShuffleNet V2 [25] and GhostNet [16], while less weights andcomputational cost.


|<sub>Model</sub>|<sub>Weights(million)</sub>|<sub>Top-1 Val Acc.(%)</sub>|<sub>FLops(million)</sub>|   
| :---: | :---: | :---: | :---: |
| <sub>[ResNet-18](https://arxiv.org/abs/1512.03385)</sub> | <sub>11</sub> | <sub>95.28</sub> | <sub>180</sub> | 
| <sub>[GhostNet](https://arxiv.org/abs/1911.11907)</sub> | <sub>5.18</sub> | <sub>93.411</sub> | <sub>141</sub> | 
| <sub>[OurNet](https://github.com/Johnny-liqiang/CellNetUML)</sub> | <sub>2.91</sub> | <sub>95.638</sub> | <sub>41.7</sub> |
| <sub>[ShuffleNet V2](https://arxiv.org/abs/1807.11164)</sub> | <sub>1.4</sub> | <sub>83.868</sub> | <sub>41</sub>|   


**Note:** 
- Speed are tested on a ETH Zurich Leonhard Cluster. 
- Performance are tested with Cellyolo preprocessing.
- This is I Chart of ournet, resnet 18, shufflenet without Cellyolo- Summary Report
<center>
    <img src="https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/I%20Chart%20of%20ournet%20on%2C%20resnet%2018%2C%20shufflenet%20withoutellyolo-%20Summary%20Report.png">
</center>


- This is Time Series Plot of Shufflenet V, ResNet18 Val, GhostNet18 V, on Sezary syndrome with cellyolo

![描述](https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/Time%20Series%20Plot%20of%20Shufflenet%20V%2C%20ResNet18%20Val%2C%20GhostNet18%20V%2C%20on%20Sezary%20with%20cellyolo.png)

### Evaluate performance on [COVID-19 Dataset](https://github.com/Johnny-liqiang/CellNetUML/tree/master/COVID19)

In order to help the medical scientists, we made this COVID-19 CT dataset. Based on initial [COVID-19  Image  Data  Collection](https://arxiv.org/abs/2003.11597),  which  contains  only  123  frontal  view  X-rays.  We  additionally  collected  data  from  newest  publications  on  European  Journal  ofRadiology,  and  collected  nearly  1583  healthy  Lung  CT/xray images  as  comparative  data  from  recently  available  resources and publications.

|<sub>Model</sub>|<sub>Weights(million)</sub>|<sub>Top-1 Val Acc.(%)</sub>|<sub>FLops(million)</sub>|   
| :---: | :---: | :---: | :---: |
| <sub>[ResNet-18](https://arxiv.org/abs/1512.03385)</sub> | <sub>11</sub> | <sub>94.389</sub> | <sub>180</sub> | 
| <sub>[GhostNet](https://arxiv.org/abs/1911.11907)</sub> | <sub>5.18</sub> | <sub>92.739</sub> | <sub>141</sub> | 
| <sub>[OurNet](https://github.com/Johnny-liqiang/CellNetUML)</sub> | <sub>2.91</sub> | <sub>94.719</sub> | <sub>41.7</sub> |
| <sub>[MobileNet V2](https://arxiv.org/abs/1801.04381)</sub> | <sub>3.4</sub> | <sub>95.38</sub> | <sub>301</sub>| 
| <sub>[Vgg11_BN](https://arxiv.org/abs/1807.11164)</sub> | <sub>13.28</sub> | <sub>87.129</sub> | <sub>132.87</sub>|
| <sub>[DenseNet121](https://arxiv.org/abs/1608.06993)</sub> | <sub>7.98</sub> | <sub>95.71</sub> | <sub>283</sub>|
| <sub>[AlexNet](http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)</sub> | <sub>60.95</sub> | <sub>0</sub> | <sub>727</sub>|
| <sub>[SqueezeNet V2](https://arxiv.org/abs/1602.07360)</sub> | <sub>--</sub> | <sub>0</sub> | <sub>40</sub>|

**Note:** 
- -- denoted un-provided.
- Speed are tested on a ETH Zurich Leonhard Cluster. 
- Performance are tested without Cellyolo preprocessing.
- This is I Chart of ournet, resnet 18, shufflenet without Cellyolo- Summary Report
<center>
    <img src="https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/ournetoncovid%20-%20Copy.png">
</center>

Our net belongs to the top list basedon the Val accuracy. Consider the fact of the higher complexity and parameters amount of other SOTA Net, our net is very competitive on classification task.

![描述](https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/ournetoncovid.png)


### Evaluate Cellyolo performance on [Sezary Syndrome Dataset](https://github.com/Johnny-liqiang/CellNetUML/tree/master/HDSS) with Saliency Map

In order to better visualizetheperformance of the cellyolo and demonstrate the necnecessity of cellyolo, we wrote saliency script to generated attention map.

| [![]()](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102).png  | [![AudioBlocks](https://dtyn3c8zjrx01.cloudfront.net/img/assets/audioblocks/images/logo.png)](http://audioblocks.com) | [![GraphicStock](http://www.graphicstock.com/images/logo.jpg)](http://graphicstock.com) |
|:---:|:---:|:---:|
| original pic: hd070916_2%20(7102).png | http://audioblocks.com | http://graphicstock.com |

| [![VideoBlocks](https://d1ow200m9i3wyh.cloudfront.net/img/assets/videoblocks/images/logo.png)](http://videoblocks.com)  | [![AudioBlocks](https://dtyn3c8zjrx01.cloudfront.net/img/assets/audioblocks/images/logo.png)](http://audioblocks.com) | [![GraphicStock](http://www.graphicstock.com/images/logo.jpg)](http://graphicstock.com) |
|:---:|:---:|:---:|
| http://videoblocks.com | http://audioblocks.com | http://graphicstock.com |

| [![VideoBlocks](https://d1ow200m9i3wyh.cloudfront.net/img/assets/videoblocks/images/logo.png)](http://videoblocks.com)  | [![AudioBlocks](https://dtyn3c8zjrx01.cloudfront.net/img/assets/audioblocks/images/logo.png)](http://audioblocks.com) | [![GraphicStock](http://www.graphicstock.com/images/logo.jpg)](http://graphicstock.com) |
|:---:|:---:|:---:|
| http://videoblocks.com | http://audioblocks.com | http://graphicstock.com |
