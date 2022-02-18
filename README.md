# LabelImg auto detection 

![ex](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)

LabelImg is a graphical image annotation tool and 'auto detection' feature is added for a ready-made weight model to detect objects automatically. You might want to check out the labelImg official github page to familiarise yourself with the original labelImg. 

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, YOLO and CreateML formats.

# Installation

clone this repo move to labelImg directory on CLI or Terminal and run 'python labelImg_custom.py'

```
python labelImg_custom.py
```

1. Choose the saving annotation type(PASCAL VOC format|YOLO|CreateML formats)

2. Click the open dir button highlightened in yellow. 
![opendir](https://user-images.githubusercontent.com/55167422/154652796-7a7cc482-bc58-44a7-b869-2740066d557a.PNG)

3. it will prompt local file loading dialog three times.
 - first dialog : image directory
 - second dialog : model file(only works for onnx)
 - last dialog : classes text 
