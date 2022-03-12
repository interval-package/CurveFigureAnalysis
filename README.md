# CurveFigureAnalysis
## 1.introduction
#### It's a project target at extracting the line figure from the picture
For example
#### Focus on get data from the pics in the papers
## 2.basic function
#### (1)read in the pics of Curve or BrokenLine.
#### (2)return the functions or points of the figure.
#### (3)figure extraction.
## 3.tech used
### basic introduce
We use three techs to detect the figure,for different compute power
#### (1)Using traditional color divide method
Basically we calc the color info and decide which color is the line color.
<img src="">
And through canny and sobel method we also get the processed pic.
Finally, we judge the quality of several pic, and return.
#### (2)Using machine learn the filter color
#### (3)Using HED to detect the edge
## 4.tutorial
#### (1)Using python API
    from LineFigure import LineFgure
    from BrokenLineFigure import BrokenLinrFigure
We define the class of "LineFigure",and subclasses such as 
BrokenLineFigure and CurveFigure for more detail process

#### (2)Using website
We build a website for 
## 5.Acknowledgement
#### contact us via e-mail: 8203200527@csu.edu.cn
