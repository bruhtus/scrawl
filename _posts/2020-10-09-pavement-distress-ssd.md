---
toc: true
layout: post
description: Detect pavement distress using single shot detector (SSD) model.
categories: [Deep Learning]
image: images/pavement-distress-ssd/logo.jpg
title: Pavement Distress Detector Using Single Shot Detector (SSD)
---
## A Brief Explanation About Single Shot Detector (SSD)

Single shot detector is a deep learning method presented by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed4, Cheng-Yang Fu, Alexander C. Berg in their research paper [SSD: Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325). There are 2 commonly used SSD model, that is, SSD300 and SSD512.
<br>
Here's a brief explanation about SSD300 and SSD512:
- SSD300: More fast.
- SSD512: More accurate.

Long story short, SSD300 is about speed. If you need speed than you should probably using SSD300 (i haven't tried the mobilenet as base network at the times to type this, so at this time knowledge SSD300 is faster than SSD512). Meanwhile, SSD512 is about accuracy. It doesn't really show up in image processing but in video processing, i notice that there's a frame rate drop while doing live object detection. To be fair, SSD300 has frame rate drop as well but it's still usable (around 7-10 frame per second) but SSD300 has frame rate around 3-5 frame per second. 
Who want to watch a video with 3 fps?? If you're that kind of person then, go ahead. You do you mate.

For the record, at that time when I try live detection, i use opencv to display live detection video. i'm not sure whether it is opencv fault or the model fault because if I save the video result, the video itself has no frame rate drop. It's weird but it happens, so let's go on with saving the video and forget about live detection (for now, until i find some way to do live detection).

So, in this project i'm not gonna make it live detection. Rather than live detections, we're gonna save the video result first and then display it later. That way it could also reduce some computational cost.

For those who still confused about live detection, to make things simpler, live detection is when you process the video, detect the object, and play the video at the same time. After you detect the object, you immediately display the frame that just recently processed and then processed the next frame. Repeat.

### Single Shot Detector (SSD) Architecture That's Used in This Project
As explained above, in this project we're gonna use SSD512. SSD512 is basically SSD with input image 512x512. The basic architecture of SSD contains 2 part, base network and extra feature layers. The base network layers are based on standard architecture used for high quality image classification (truncated before classification layers). The extra feature layers used for multi-scale feature maps for detection and convolutional predictors for detection.

Here is an architecture single shot detector that used in this project (made this with [NN architecture maker](http://alexlenail.me/NN-SVG/AlexNet.html)):
![]({{site.baseurl}}/images/pavement-distress-ssd/ssd-architecture.png)<br>
Information:
1. Input image.
2. Base Network (truncated before classification layers).
3. Layer 6 and layer 7 of base network (from fully-connected layer turned into classification layer).
4. Extra feature layers.
5. Collection of boxes and scores.

#### Base Network
The base network used in this project is Visual Geometry Group (VGG). I chose VGG because of transfer learning capability so that i could have a good result with small dataset. To be more specific, in this project i used VGG16, here is a brief explanation of each layers:
1. In the first layer, there's a convolutional process with kernel filter 3x3 and stride (total shift filter per pixel) 1 pixel. That process repeat 2 times and then did some max pooling with kernel filter 2x2 and stride 2 pixel.
2. In the second layer until fourth layer, the model did the same thing as in first layer.
3. The difference was in fifth layer. In fifth layer, the convolution process still the same as the other four layers but the max pooling process was different from the other four layers. The max pooling process used kernel filter 3x3 with stride 1 pixel with padding (adding zero value around pixel image) 1. You can check the illustration below to understand the process of max pooling with kernel filter 3x3, stride 1, and padding 1. <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/max-pooling-illustration.png)

And here's a VGG16 after truncated from classification layers: <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/truncated-vgg16.jpg)

If you want to calculate the result from max polling, you can use this equation: <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/max-pooling-equation.png)<br>
Information:
1. kernel_size, stride, padding, and dilation can be 1 integer (in this case, the value for height and width are the same) or 2 integer (in this case, the first integer is height and the second integer is width).
2. For more info you can see [pytorch page](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).

Here's some example of max pooling calculation with input 32x32, kernel filter 3x3, stride 1, padding 1, and dilation 1: <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/max-pooling-calculation.png)

#### Layer 6 and Layer 7
After feature extraction process in base network, the next layers is to change layer 6 and 7 of base network from fully-connected into convolutional layer with subsample parameters from fully-connected 6 (fc6) and fully-connected 7 (fc7). The convolution operation used in layer 6 and layer 7 is atrous convolution, you can see atrous convolution shift below: <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/atrous-convolution.png)

With atrous convolution we can expand area of observation for feature extraction while maintaning the amount of parameters fewer than traditional convolution operation.

#### Extra Feature Layers
Extra feature layers is a prediction layers. In this layer, the model predict the object using default box. Default box is a box with various aspect ratio in every location of feature maps with different size. You can see an example of default box below (from SSD research paper): <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/default-box.png)

In the last layer is a collection of default boxes which closer to ground truth box with confidence score from that default boxes.

---

## Take A Video (Training Video and Testing Video)

In this part, i'm gonna explain about the video used in this project. The camera configuration, the place where the video taken, the camera angle and height from the road.

The place where the video taken was in Surabaya, at Kertajaya Indah Timur IX, Kertajaya Indah Timur X, and Kertajaya Indah Timur XI. The camera angle was perpendicular(?) with the road (90 degrees) and the camera position from the road was 200 cm.

There're 7 video taken, 3 for training and 4 for testing. The format of the video was `*.mp4`. You can check the location partition of the video taken below: <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/video-taken.jpg)<br>
The black block is for testing and the white block is for training. You can check the position of the camera below: <br>
![]({{site.baseurl}}/images/pavement-distress-ssd/camera-position.png)

---

## Setting Up The Config File

For more detailed configuration please check [develop guide by Congcong Li](https://github.com/lufficc/SSD/blob/master/DEVELOP_GUIDE.md). In this post i'm gonna explain it the easiest way.

### Basic Configuration
To make things easier, copy the format dataset you want. For example, in this project i want to use COCO dataset format. Then, i copied the `coco.py` in the path `ssd/data/datasets/` and rename it to `my_dataset.py`. After that, edit the class names for your classification class. In this project, the class i'm gonna use is alligator crack, longitudinal crack, transverse crack, and pothole. Also, don't forget to change the class `COCODataset` to `MyDataset`.

The next step is to add those configuration to `__init__.py` in ssd/data/datasets/. For example:
```python
from .my_dataset import MyDataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'MyDataset': MyDataset,
}
```

Another next step is to add the path of your datasets and anotations to the `path_catlog.py` in `ssd/config/`. For example:
```python
import os

class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'my_custom_train_dataset': {
            "data_dir": "train",
            "ann_file": "annotations/train.json"
        },

        'my_custom_validation_dataset': {
            "data_dir": "validation",
            "ann_file": "annotations/validation.json"
        },
    }
    
    @staticmethod
    def get(name):
        if "my_custom_train_dataset" in name:
            my_custom_train_dataset = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                    data_dir = os.path.join(my_custom_train_dataset, attrs['data_dir']),
                    ann_file = os.path.join(my_custom_train_dataset, attrs['ann_file']),
            )
            return dict(factory="MyDataset", args=args)

         if "my_custom_test_dataset" in name:
            my_custom_train_dataset = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                    data_dir = os.path.join(my_custom_train_dataset, attrs['data_dir']),
                    ann_file = os.path.join(my_custom_train_dataset, attrs['ann_file']),
            )
            return dict(factory="MyDataset", args=args)
```

And finally, for the `*.yaml` file for configuration i copied `vgg_ssd512_coco_trainval35k.yaml` in `configs` folder and rename it to `config.yaml`. What i changed from that file was the train and test (or more like validation) like in `path_catlog.py`, the batch size, and num_classes. I changed batch size because my laptop gpu only capable of 4 batch size. Here's an example:
```python
Model:
    num_classes: 5 #the __background__ counted
    ...
    DATASETS:
        TRAIN: ("my_custom_train_dataset", )
        TEST: ("my_custom_test_dataset", )
    SOLVER:
        ...
        BATCH_SIZE: 4
        ...
    
    OUTPUT_DIR: 'outputs/ssd_custom_coco_format'
```
You don't need to create folder `ssd_custom_coco_format`, when the training begin the folder gonna created automatically (if the folder didn't exist).

### Validation Configuration
First of all, copy `coco` folder in `ssd/data/datasets/evaluation/` and rename it to `my_dataset`. Rename the `def coco_evaluation` to `def my_dataset_evaluation` in file `__init__.py`. After that, add folder `my_dataset` to file `__init__.py` in `ssd/data/datasets/evaluation/`. For example:
```python
from ssd.data.datasets import VOCDataset, COCODataset, MyDataset
...
from .my_dataset import my_dataset_evaluation

def evaluate(dataset, predictions, output_dir, **kwargs):
    ...
    elif isinstance(dataset, MyDataset);
        return my_dataset_evaluation(**args)
    else:
        raise NotImplementError
```

## Training Preparation

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Here We Go, It's Training Time!

![]({{ site.baseurl }}/images/logo.png "fast.ai's logo")

## Testing Preparation

You can format text and code per usual 

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

Formatting text as shell commands:

```shell
echo "hello world"
./some_script.sh --option "value"
wget https://example.com/cat_photo1.png
```

Formatting text as YAML:

```yaml
key: value
- another_key: "another value"
```


## Go Get Them (The Pavement Distresses)! It's Testing Time!

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |


## A Brief Showcase and Explanation of The Results

{% twitter https://twitter.com/jakevdp/status/1204765621767901185?s=20 %}


## Future Suggestion



[^1]: This is the footnote.

