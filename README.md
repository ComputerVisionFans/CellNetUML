
This is the official implementation with *training* code for Cell Morphology Based Diagnosis of Cancer usingConvolutional Neural Networks: CellNet. For technical details, please refer to:

**Cell Morphology Based Diagnosis of Cancer usingConvolutional Neural Networks: CellNet** <br />
[Qiang Li](https://www.linkedin.com/in/qiang-li-166362143/)\*, [Corin](https://biol.ethz.ch/en/the-department/people/person-detail.MTkyNzk5.TGlzdC80NjAsOTIzMDMxMjIy.html)\* <br />
**Paper In preparation for ICPR2020** <br />
**[[Software Report](https://drive.google.com/file/d/1_lVADVGAhkRG2qvzbhACQ5JqrCPpYRwM/view?usp=sharing)] [[CellNetSoftware Video](https://drive.google.com/open?id=1V-VDtiOJv1WI5jVSQZqpj-6AKPFUBexX)] [[Research Grant Page](https://ethz.ch/en/studies/non-degree-courses/exchange-and-visiting-studies/programmes/exchange-programmes/idea-league.html)]** <br />


This is the Boxplot of resnet18 , Ournet, ghostnet on cifar without cellyolo, as it illustrated that, even without Cellyolo preprocessing, our net already achieved the best performance. Due to the fact, every image from this dataset is 32*32 pixel image, it's getting hard to train a well segmentor by cellyolo to filter out the other artifacts in the image.


<center>
    <img src="https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/Boxplot%20of%20resnet18%20%2C%20OurGhostRe%2C%20ghostnet%20on%20cifar%20without%20cellyolo%20-%20Summary%20Report.png">
</center>



![描述](https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/Time%20Series%20Plot%20of%20ghost%20net%20on%2C%20resnet18%20on%20%2C%20our%20net%20on%20pneumonia%20dataset%20without%20cellyolo.png)




<center>
    <img src="https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/I%20Chart%20of%20ournet%20on%2C%20resnet%2018%2C%20shufflenet%20withoutellyolo-%20Summary%20Report.png">
</center>





![描述](https://github.com/Johnny-liqiang/CellNetUML/blob/master/paperimage/plot/Time%20Series%20Plot%20of%20Shufflenet%20V%2C%20ResNet18%20Val%2C%20GhostNet18%20V%2C%20on%20Sezary%20with%20cellyolo.png)




