# YOLO9000:
Better, Faster, Stronger

Author: Joseph Redmon, Ali Farhadi
Journal: Computer Vision and Pattern Recognition
PDF: YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/YOLOv2.pdf
Published Date: 2016년 12월 25일
keyword: Object Detection
link: https://arxiv.org/pdf/1612.08242.pdf

# Abstract

We introduce YOLO9000, a state-of-the-art, real-time object detection system that can detect over 9000 object categories.

we propose various improvements to the YOLO detection method.

The improved model, YOLOv2, is state-of-the-art on standard detection tasks like PASCAL VOC and COCO.

Using this method we train YOLO9000 simultaneously on the COCO detection dataset and the ImageNet classification dataset.

YOLO can detect more than just 200 classes; it predicts detections for more than 9000 different object categories. And it still runs in real-time.

# 1. Introduction

Our method uses a hierarchical view of object classification that allows us to combine distinct datasets together.

We also propose a joint training algorithm that allows us to train object detectors on both detection and classification data.

Our method leverages labeled detection images to learn to precisely localize objects while it uses classification images to increase its vocabulary and robustness.

Using this method we train YOLO9000, a real-time object detector that can detect over 9000 different object categories.

# 2. Better

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled.png)

YOLO makes a significant number of localization errors.

YOLO has relatively low recall compared to region proposal-based methods.

Thus we focus mainly on improving recall and localization while maintaining classification accuracy.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%201.png)

**Batch Normalization**

By adding batch normalization on all of the convolutional layers in YOLO we get more than 2% improvement in mAP.

With batch normalization we can remove dropout from the model without overfitting.

**High Resolution Classifier**

For YOLOv2 we first fine tune the classification network at the full 448 x 448 resolution for 10 epochs on ImageNet.

This high resolution classification network gives us an increase of almost 4% mAP.

**Convolutional With Anchor Boxes**

We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxes.

We also shrink the network to operate on 416 input images instead of 448x448.

we want an odd number of locations in our feature map.

we get an output feature map of 13 x 13.

we move to anchor boxes we also decouple the class prediction mechanism from the spatial location and
instead predict class and objectness for every anchor box.

Using anchor boxes we get a small decrease in accuracy

69.5 mAP with a recall of 81%.

With anchor boxes our model gets 69.2 mAP with a recall of 88%. 

Even though the mAP decreases, the increase in recall means that our model has more room to improve.

**Dimension Clusters.**

issue 1. the box dimensions are hand picked.

we run k-means clustering on the training set bounding boxes to automatically find good priors.

If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%202.png)

We choose k = 5 as a good tradeoff between model complexity and high recall.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%203.png)

If we use 9 centroids we see a much higher average IOU. This indicates that using k-means to generate our bounding box starts the model off with a better representation and makes the task easier to learn.

**Direct location prediction**

issue 2. model instability, especially during early iterations.

Most of the instability comes from predicting the (x, y) locations for the box.

In region proposal networks the network predicts values tx and ty and the (x, y) center coordinates are calculated as : 

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%204.png)

We use a logistic activation to constrain the network’s predictions to fall in this range.

The network predicts 5 coordinates for each bounding box, tx, ty, tw, th, and to.

If the cell is offset from the top left corner of the image by (cx, cy) and the bounding box prior has width and height pw, ph, then the predictions correspond to:

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%205.png)

Since we constrain the location prediction the parametrization is easier to learn, making the network
more stable.

Using dimension clusters along with directly predicting the bounding box center location improves
YOLO by almost 5% over the version with anchor boxes.

**Fine-Grained Features**

This modified YOLO predicts detections on a 13 x 13 feature map.

We take a different approach, simply adding a passthrough layer that brings features from an earlier layer at 26 x 26 resolution.

This gives a modest 1% performance increase.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%206.png)

**Multi-Scale Training.**

With the addition of anchor boxes we changed the resolution to 416x416.

We want YOLOv2 to be robust to running on images of different sizes so we train this into the model.

Every 10 batches our network randomly chooses a new image dimension size.

following multiples of 32: {320, 352, ..., 608}.

Thus the smallest option is 320 x 320 and the largest is 608 x 608.

This regime forces the network to learn to predict well across a variety of input dimensions.

YOLOv2 offers an easy tradeoff between speed and accuracy

# 3. Faster

We want detection to be accurate but we also want it to be fast.

**Darknet-19**

We propose a new classification model to be used as the base of YOLOv2.

we use global average pooling to make predictions as well as 1 x 1 filters to compress the feature representation between 3 x 3 convolutions.

We use batch normalization to stabilize training, speed up convergence, and regularize the model.

Our final model, called Darknet-19, has 19 convolutional layers and 5 maxpooling layers.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%207.png)

# 4. Stronger

ImageNet has more than a hundred breeds of dog, including “Norfolk terrier”, “Yorkshire terrier”, and “Bedlington terrier”.

Using a softmax assumes the classes are mutually exclusive.

We could instead use a multi-label model to combine the datasets which does not assume mutual exclusion.

**Hierarchical classification.** 

ImageNet labels are pulled from WordNet, a language database that structures concepts and how they relate.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%208.png)

To compute the conditional probabilities our model predicts a vector of 1369 values and we compute the softmax over all sysnsets that are hyponyms of the same concept, see Figure 5.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%209.png)

**Dataset combination with WordTree.**

We can use WordTree to combine multiple datasets together in a sensible fashion.

Figure 6 shows an example of using WordTree to combine the labels from ImageNet and COCO.

![Untitled](YOLO9000%20Better,%20Faster,%20Stronger%20f61dbbac5f8f4a6994cb7634f235533e/Untitled%2010.png)

**Joint classification and detection.**

Using this dataset we train YOLO9000. We use the base YOLOv2 architecture but only 3 priors instead of 5 to limit the output size.

# 5. Conclusion

Furthermore, it can be run at a variety of image sizes to provide a smooth tradeoff between speed and accuracy.

YOLO9000 is a real-time framework for detection more than 9000 object categories by jointly optimizing detection and classification.