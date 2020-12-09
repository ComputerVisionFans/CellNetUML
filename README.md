# CellNet and CellNet Software
This is the official implementation with *training* code for Thesis: Cell Morphology Based Diagnosis of Cancer usingConvolutional Neural Networks: CellNet. For technical details, please refer to:

**Cell Morphology Based Diagnosis of Cancer usingConvolutional Neural Networks: CellNet** <br />
[Qiang Li](https://www.linkedin.com/in/qiang-li-166362143/)\*, [Otesteanu Corin](https://biol.ethz.ch/en/the-department/people/person-detail.MTkyNzk5.TGlzdC80NjAsOTIzMDMxMjIy.html)\* [Manfred Claassen](https://imsb.ethz.ch/research/claassen/people/manfredclaassen.html)\*<br />
**Paper In preparation** <br />
**[[Software Report](https://drive.google.com/file/d/1fznzulJ-K8Fj7SNC3g89RFZLNUl39vL0/view?usp=sharing)] [[CellNetSoftware Video](https://drive.google.com/file/d/1wCXke7iyolk2AkwOOM3Fu1uONrznqUEs/view?usp=sharing)] [[Research Grant Page](https://ethz.ch/en/studies/non-degree-courses/exchange-and-visiting-studies/programmes/exchange-programmes/idea-league.html)]** <br />

## Results
These are the reproduction results from this repository. All the training/testing lsg log file on ETH Zurich leonhard cluster can be downloaded from our [lsf file](https://github.com/Johnny-liqiang/CellNetUML/tree/master/training%20log%20file%20for%20verification) and all original data for generating those data analyse graph can be downloaded from [all data file](https://github.com/Johnny-liqiang/CellNetUML/blob/master/master%20thesis%20related%20graph%20data%20%20%20(updated%20until%2025052020).xlsx)

### Evaluate performance on [CiFar10](https://www.cs.toronto.edu/~kriz/cifar.html)
This is the Boxplot of resnet18, Ournet, ghostnet on cifar without Cellyolo. Due to the fact that every image from this dataset is 32*32 pixel image, it's getting hard to train a well segmentor by cellyolo to filter out the other artifacts in the image. As it illustrated that, even without Cellyolo preprocessing, our net already achieved the best performance. 

|<sub>Model</sub>|<sub>Weights(million)</sub>|<sub>Top-1 Val Acc.(%)</sub>|<sub>FLops(million)</sub>|   
| :---: | :---: | :---: | :---: |
| <sub>[VGG-16](https://arxiv.org/abs/1409.1556)</sub> | <sub>15</sub> | <sub>93.6</sub> | <sub>313</sub>|        
| <sub>[ResNet-18](https://arxiv.org/abs/1512.03385)</sub> | <sub>11</sub> | <sub>91.96</sub> | <sub>180</sub> | 
| <sub>[GhostNet](https://arxiv.org/abs/1911.11907)</sub> | <sub>5.18</sub> | <sub>91.45</sub> | <sub>141</sub> | 
| <sub>[OurNet](https://github.com/Johnny-liqiang/CellNetUML)</sub> | <sub>2.91</sub> | <sub>92.45</sub> | <sub>41.7</sub> |


CIFAR-10  dataset  consists of 60,000 32 × 32 color images in 10 classes, with 50,000 training images and 10,000 test images.  A common  data  augmentation  scheme  including random crop and mirroring is adopted as well.

**Note:** 
- Speed are tested on a ETH Zurich Leonhard Cluster. 
- You will see Cellyolo and ghostresNet in several places, please donot be frustrated, Cellyolo = AttentionNet in the paper, and ghostresNet = CellNet in the paper, just nickname:)!. 

<p align="center">
  
  <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Cifar-12-06-2020.png" width="500" alt="Comparison  of  state-of-art  methods  on  CIFAR10  Dataset">
  
</p>


### Evaluate performance on [Pneumonia Dataset](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site)

On  benchmark  pneumonia  dataset,  the  Pneumonia/Normal classification val accuracy of our Net converges into nearly 91.785% better than Ghost Net and ResNet18, In addition, after around 80 epochs the accuracy of our Net converged, comparing to  [Inception  V3](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site)  after 7000 epochs reaches 88.0%.

|<sub>Model</sub>|<sub>Weights(million)</sub>|<sub>Top-1 Val Acc.(%)</sub>|<sub>FLops(million)</sub>|   
| :---: | :---: | :---: | :---: |
| <sub>[InceptionV3](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site)</sub> | <sub>23.81</sub> | <sub>88</sub> | <sub>540</sub>|  
| <sub>[ResNet-18](https://arxiv.org/abs/1512.03385)</sub> | <sub>11</sub> | <sub>87.50</sub> | <sub>180</sub> | 
| <sub>[GhostNet](https://arxiv.org/abs/1911.11907)</sub> | <sub>5.18</sub> | <sub>88.69</sub> | <sub>141</sub> | 
| <sub>[OurNet](https://github.com/Johnny-liqiang/CellNetUML)</sub> | <sub>2.91</sub> | <sub>91.78</sub> | <sub>41.7</sub> |
 

<p align="center">
  <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Pneumonia_TimeSeries-1.png" width="500" alt="Comparison  of  state-of-the-art  methods  for  training  onPneumonia Dataset">
</p>


### Evaluate performance on [Sezary Syndrome Dataset](https://github.com/Johnny-liqiang/CellNetUML/tree/master/HDSS)

ResNet18  [17]  and  ShuffleNetv2  [25]  were  verified  so  far the most representative best performance on Sezary Syndrome Dataset. But Our* Net can achieve higher classification perfor-mance  (e.g.  95.638%  top-1  accuracy  )  than  ResNet  18  [17], ShuffleNet V2 [25] and GhostNet [16], while less weights and computational cost.


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

<p align="center">
  <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Sesary_TimeSeries-12-06-2020.png" width="500" alt="Sezary Syndrome-Dataset.">
</p>



### Evaluate performance on [COVID-19 Dataset](https://github.com/Johnny-liqiang/CellNetUML/tree/master/COVID19)

In order to help the medical scientists, we made this COVID-19 CT dataset. Based on the initial [COVID-19  Image  Data  Collection](https://arxiv.org/abs/2003.11597), which contains only 123 frontal view X-rays. We also collected data from the newest publications on the European Journal of Radiology and collected nearly 1583 healthy Lung CT/Xray images as comparative data from recently available resources and publications.

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

<p align="center">
  <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/COVID-19_TimeSeries-1.png" width="700" alt=" COVID-19  Dataset">
</p>

Comparison of  state-of-art methods for training on   COVID-19 Dataset.   Our models' weights are 2.91  million,  comparing  toDenseNet121 7.98 million of weights,  MobileNet V2 3.4 million of weights, and 301 million of FLOPs; considering the higher complexity and parameter amount of other  SOTA  Nets,  our  Net is very competitive on classification tasks for the biomedical dataset.




### Evaluate Cellyolo performance on [Sezary Syndrome Dataset](https://github.com/Johnny-liqiang/CellNetUML/tree/master/HDSS) with Saliency Map

To better visualize the performance of the cellyolo and demonstrate the necessity of cellyolo, we wrote a saliency script to generate an attention map. ResNet18 puts more attention on the outside of ROI, while VGG and our Net focus more on ROI. Cellyolo is playing a vital role in eliminating the artifacts, enforcing the models more focus on the cell itself.

**Note:** 
- For more attention maps see [saliencymap folder](https://github.com/Johnny-liqiang/CellNetUML/tree/master/saliencymap).



| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd070916_2%20(7102).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd070916_2%20(7102).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7102)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd070916_2 (7102).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |




| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd070916_2%20(7558).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd070916_2%20(7558).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd070916_2%20(7558)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd070916_2 (7558).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |




| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd1%20(3697).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd1%20(3697).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(3697)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd1 (3697).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd1%20(4550).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd1%20(4550).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4550)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd1 (4550).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd17_5%20(1876).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd17_5%20(1876).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd17_5%20(1876)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd17_5 (1876).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |





| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd1%20(4400).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd1%20(4400).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd1%20(4400)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd1 (4400).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd3%20(1).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/hd3%20(1).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/hd3%20(1)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: hd3 (1).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss2_8%20(117).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss2_8%20(117).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(117)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: ss2_8 (117).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss1_2%20(270).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss1_2%20(270).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss1_2%20(270)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: ss1_2 (270).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss2_8%20(142).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss2_8%20(142).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(142)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: ss2_8 (142).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |


| [![original pic](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468).png)  | [![aftercellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss2_8%20(468).png)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/cellyolo/output/ss2_8%20(468).png) | [![ournetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)ournetWithcellyolo.jpg) | [![ournetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)ournetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)ournetWithoutcellyolo.jpg) |[![resnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)resnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)resnetWithcellyolo.jpg) | [![resnetwithoutcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)resnetWithoutcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)resnetWithoutcellyolo.jpg) | [![vggnetwithcellyolo](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)vggnetWithcellyolo.jpg)](https://github.com/Johnny-liqiang/CellNetUML/blob/master/saliencymap/ss2_8%20(468)vggnetWithcellyolo.jpg) | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original pic: ss2_8 (468).png | After Cellyolo segmentation | Ournet with cellyolo | Ournet without cellyolo | Res18 with cellyolo | Res18 without cellyolo | Vgg16 with cellyolo |

## The generalization  performance with our best weight(with/without finetuning)
### Non-cerebriform dataset

#### CellNet before finetuning
Prediction with our CellNet best weight trained so far on Non-cerebriform dataset, As shown in the figure, the TP and TN achieved the general highest score on HD/SS with more considerable image amount. Moreover, average accuracy up to 99.53%-96.51% among HD image, and average accuracy achieved 92.19%-98.78% among SS image, but there is some small folder obtain 38.29%-37.48% on SS1 and SS2, 40.17% in SS6_B folder as well.
<center>
    <img src="https://github.com/Johnny-liqiang/CellNetUML/blob/master/trainsfer%20learning%20on%20Non-ceribriform/before%20finetuning.JPG">
</center>

#### CellNet after finetuning
After further finetuning, basically using best weight trained so far + new subset of Non-cerebriform, and set mini batch=679, trained around 100 epochs. We test the performance again. As shown, the accuracy is improved with SS1 and SS2 and SS6_B folder surprisingly up to 64.34%, 82.64%, and 96.91%.

<center>
    <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Non-ceribriform.jpg">
</center>

This is the comparison between Cellnet and  ResNet18 on the Non-cerebriform dataset with finetuning. As illustrated, our net has comparable Acc. even some higher on some folder.



### cerebriform dataset
Prediction with our CellNet best weight trained so far on the cerebriform dataset, As shown in the figure, the TP and TN achieved comparable accuracy(in %) with resnet18.

<center>
    <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Ceribriform.jpg">
</center>


Now our software upload on nash cloud as well, and support pretrained_weight further training, and all the prediction lsg files you can check here: [lsg file for you to check](https://github.com/Johnny-liqiang/CellNetUML/tree/master/trainsfer%20learning%20on%20Non-ceribriform)

### [Table results of  Ceribriform dataset /Ceribriform dataset ](https://github.com/Johnny-liqiang/CellNetUML/blob/master/master%20thesis%20related%20graph%20data%20%20%20(updated%20until%2005072020).xlsx)



## How to train with your data
You want to have it try by your own dataset with our cellnet. No problem! These are all the [commands](https://github.com/Johnny-liqiang/CellNetUML/blob/master/The%20%20commond%20so%20far%20to%20runing%20the%20project.docx)

## Take a look at our CellNet software framework and Our CellNet won Top AI Camp [Deecamp2020](https://www.chuangxin.com/index.php/home/aboutus/news_detail/id/412.html) Medical Track 2nd place
With the help of the power of Qt and the high efficiency of Python, using PyQt/PySide for desktop development will be a wonderful plus for demonstrating our excellent software. The current Qt/PyQt/PySide based GUI development common development methods are list follow:  QWidget + QSS, QtWebkit+ HTML + CSS + js and Qt Quick.  All these three technologies can efficiently and quickly  develop  the  crossplatform  desktop  software.   Qt’s  formal  development  method  is  Qt  Quick,  which  uses  the  JSON  like  language  qml  for  rapid development.  It is easy to learn, expansible, and wildly used in Ubuntu, LinuxDeepin, and other Linux desktop application development.  It enables the developer for a rapid development framework and putting more effort into amplifying the  corresponding  business  logic  and  easy  to  build  the  framework  prototypes quickly.  

<center>
    <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Capture.PNG">
</center>

The Proposed software structure diagram. To better demonstrate our model’s diagnostic performance, we selected the classic medical bench-mark datasets from competitions on Kaggle, such as the melanoma dataset, the diabetic retinopathy dataset, the actinic keratosis, vascular lesion dataset dermatofibroma dataset, squamous cell carcinoma dataset. Meanwhile, we selected nearly 11 representative classification networks, enable users to choose the di-agnostic network that fits their customer dataset. Besides, we inherit the computer vision classification network and the classic classification network of NLP. We develop desktop applications and open APIs to facilitate a better user experience, and ETH Leonhard and Megengine jointly provide our computing power.

<center>
    <img src="https://github.com/Johnny-liqiang/thesis-template-master_rwth/blob/master/thesis-template-master/images/Pages%20from%20Qiang%20Li%20Application%20certificate%20documents_1.jpg">
</center>



## License
All Software copy right licensed by QiangLi

