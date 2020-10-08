---
toc: true
layout: post
description: A minimal example of using markdown with fastpages.
categories: [Deep Learning]
title: Pavement Distress Detector Using Single Shot Detector (SSD)
---
# Pavement Distress Detector Using Single Shot Detector (SSD)

## A Brief Explanation About Single Shot Detector (SSD)

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then a space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

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

