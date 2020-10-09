---
toc: true
layout: post
description: Detect pavement distress using single shot detector (SSD) model.
categories: [Deep Learning]
title: Pavement Distress Detector Using Single Shot Detector (SSD)
---
## A Brief Explanation About Single Shot Detector (SSD)

Single shot detector is a deep learning method presented by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed4, Cheng-Yang Fu, Alexander C. Berg in their research paper [SSD: Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325). There are 2 commonly used SSD model, that is, SSD300 and SSD512.
<br>
Here's a brief explanation about SSD300 and SSD512:
- SSD300: More fast
- SSD512: More accurate

Long story short, SSD300 is about speed. If you need speed than you should probably using SSD300 (i haven't tried the mobilenet as base network at the times to type this, so at this time knowledge SSD300 is faster than SSD512). Meanwhile, SSD512 is about accuracy. It doesn't really show up in image processing but in video processing, i notice that there's a frame rate drop while doing live object detection. To be fair, SSD300 has frame rate drop as well but it's still usable (around 7-10 frame per second) but SSD300 has frame rate around 3-5 frame per second. 
Who want to watch a video with 3 fps?? If you're that kind of person then, go ahead. You do you mate.

For the record, at that time when I try live detection, i use opencv to display live detection video. i'm not sure whether it is opencv fault or the model fault because if I save the video result, the video itself has no frame rate drop. It's weird but it happens, so let's go on with saving the video and forget about live detection (for now, until i find some way to do live detection).

So, in this project i'm not gonna make it live detection. Rather than live detections, we're gonna save the video result first and then display it later. That way it could also reduce some computational cost.

For those who still confused about live detection, to make things simpler, live detection is when you process the video, detect the object, and play the video at the same time. After you detect the object, you immediately display the frame that just recently processed and then processed the next frame. Repeat.

## Take A Video (Training Video and Testing Video)

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Setting Up The Config File

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

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

