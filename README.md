# Object_Detection_With_Yolov4
Object Detection with Yolov4 implementation

## What is YOLO algorithm?
Yolo is the state of art object detection algorithm & it is so fast that it has become a almost standard way of detecting object in the field of computer vision. In 2015 `YOLO` outperformed all the previous object detection algorithm.

The full form is , **You Only Look Once**

Let's say we are working on image classification where we decide the image is of a dog or a person. Let's have a look,

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F5a72c4a54d73937c87463b73bc147776%2Fdog.jpg?generation=1680362922991837&alt=media)

It is pretty simple,  we will say dog is 1 and person is 0 or
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F762d7b396a4276d3feab7428a4350656%2Fout.PNG?generation=1680363097839436&alt=media)

When we talk about **Object Localization** , then, we are not talking only about which class this is but also telling about bounding box or the position of the object within the image. 

### Object Localization

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F9da8df84ef23afb28595e2412eb8849e%2Fdog.jpg?generation=1680364239198153&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F2ea26eb1811347d33584513aaf930188%2Fout2.PNG?generation=1680363370722053&alt=media)

How we will do that actually?
In terms of neural network output we can have a vector like this- 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F3f094cdbbfce0546200b5fa02744b015%2Fvector.PNG?generation=1680363908597352&alt=media)

Here, P<sub>c</sub> is the probability of the class, if there is a dog or person then there number will be 1, or if  there is no dog or person then there number will be 0

B<sub>x</sub>, B<sub>y</sub> is bounding box and they are the co-ordinate of the center which indicated yellow circle in the picture.  B<sub>w</sub>,  B<sub>h</sub>
is the width and height of the box. 
C<sub>1</sub> is for dog class and C<sub>2</sub> is person class.

So now we can train a neural  network to classify the object as well as bounding box.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F6cff8f73dbd72d76335919c59d458ff3%2Fcnn.PNG?generation=1680366271602074&alt=media)

I am just showing 2 images but there will be a thousand images & for each image since its a supervised learning problem we need to give a bounding boxes. Then we need to convert it into this vector like above image. Here image is X_train and vector is y_train with vector of size 7.  This is just for one object classification but when we have multiple object in one image then it is hard to to determine the size of neural network is hard.  So, we have to do something.

Here we have a image like below and have two bounding box - 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F55f7c0a844b5da166f0acd6ea704c8dc%2Fyolo1.PNG?generation=1680367987136128&alt=media)

YOLO algorithm will divide it will divide image into this kind of grid cells.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F614615098d22eb577291dc51ce8a0744%2Fgrid.PNG?generation=1680368116270739&alt=media)

So, I am using 4 by 4 grid here. It could be 3 by 3 or 11 by 11. For each of grid cell we can encode or we can come up with a vector that we say in previous.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F5ad293ad27fbaeca54f52c35656ec8f1%2Fmanyyoylo.PNG?generation=1680368667380344&alt=media)

So now we have an input image and it’s corresponding target vector. Using the above example, our model will be trained as follows:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F9494541%2F6fe6cb126e6f5dcbde091d5d72ff7e38%2Fprocess.PNG?generation=1680369486725769&alt=media)

We will run both forward and backward propagation to train our model. During the testing phase, we pass an image to the model and run forward propagation until we get an output y. In order to keep things simple, I have explained this using a 4 X 4 grid here, but generally in real-world scenarios we take larger grids (perhaps 19 X 19).

Even if an object spans out to more than one grid, it will only be assigned to a single grid in which its mid-point is located. We can reduce the chances of multiple objects appearing in the same grid cell by increasing the more number of grids (19 X 19, for example).
SO, actually YOLO divides an input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and how accurate it thinks the predicted box is.

YOLO predicts multiple bounding boxes per grid cell. At training time, we only want one bounding box predictor to be responsible for each object. YOLO assigns one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at forecasting certain sizes, aspect ratios, or classes of objects, improving the overall recall score.

One key technique used in the YOLO models is non-maximum suppression (NMS). NMS is a post-processing step that is used to improve the accuracy and efficiency of object detection. In object detection, it is common for multiple bounding boxes to be generated for a single object in an image. These bounding boxes may overlap or be located at different positions, but they all represent the same object. NMS is used to identify and remove redundant or incorrect bounding boxes and to output a single bounding box for each object in the image.


### Why the YOLO algorithm is important
**YOLO algorithm is important because of the following reasons:**

**Speed:** This algorithm improves the speed of detection because it can predict objects in real-time.
**High accuracy:** YOLO is a predictive technique that provides accurate results with minimal background errors.
**Learning capabilities:** The algorithm has excellent learning capabilities that enable it to learn the representations of objects and apply them in object detection.


Ref: [YOLO](https://www.v7labs.com/blog/yolo-object-detection#:~:text=YOLO%20divides%20an%20input%20image,confidence%20scores%20for%20those%20boxes.) 

Ref: [YOLO working process](https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/)
