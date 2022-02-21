# LabelImg auto detection 

![ex](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)

LabelImg is a graphical image annotation tool and 'auto detection' feature is added for a ready-made weight model to detect objects automatically. You might want to check out the labelImg official github page if you are not familiar with the original labelImg. 

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, YOLO and CreateML formats.

# Installation

clone this repo and move to labelImg directory on CLI/Terminal and run 'python labelImg_custom.py'

```
python labelImg_custom.py
```
# How to use

1. Choose the saving annotation type(PASCAL VOC format|YOLO|CreateML formats)

2. Click the open dir button highlightened in yellow. 
![opendir](https://user-images.githubusercontent.com/55167422/154652796-7a7cc482-bc58-44a7-b869-2740066d557a.PNG)

3. It will prompt local file loading dialogs three times.
 - first dialog : image directory
 - second dialog : model file(works for onnx only)
 - last dialog : classes list txt file(one class name per line)
 
 4. It will create annotation files in the image directory in a chosen annotation type and load onto the tool with auto detected bounding boxes. The rest features(edit, save, delete etc.) on labelImg are still supported for the newly detected bboxes.  
