import cv2
import glob 
import numpy as np
from pathlib import PurePath
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree
from libs.labelFile_custom import LabelFile

def jsondet(source, onnx, class_path):
    net = cv2.dnn.readNet(onnx, config=None, framework=None)
    imgformat = ['jpg', 'png', 'psd', 'tiff', 'gif', 'jpeg']
    with open(class_path) as f:
        classes = f.read().split('\n')
    # rubbish.txt = ['Styrofoam_Box', 'Styrofoam_Buoy', 'Styrofoam_Piece', 'PET_Bottle', 'Glass', 'Metal', 'Plastic_Buoy', 'Plastic_Buoy_China', 'Plastic_ETC', 'Net', 'Rope'] 
    # ES.txt = ['Echi', 'Star'] 
    jsonoutput = {}
    for image in glob.glob(source):
        p = PurePath(image)
        if p.parts[-1].split('.')[1] not in imgformat:
            if p.parts[-1].split('.')[1] not in ['json', 'xml', 'txt']:
                print('Check the image file type.', p.parts[-1])
            continue
        jsonfilename = p.parts[-1].split('.')[0] + '.json'
        jsonoutput["image"] = p.parts[-1]
        annotations = []
        img = cv2.imread(image)
        blob = cv2.dnn.blobFromImage(img, scalefactor= 1/255, size = (640, 640), swapRB=False)
        net.setInput(blob)
        prob = net.forward()
        class_ids = []
        confidences = []
        boxes = []
        originalxpcnt = img.shape[1]/640 #원본에서 resize된 너비 비율
        originalypcnt = img.shape[0]/640 #원본에서 resize된 높이 비율
        for box in prob[0]:
            if box[4] > 0.5:
                scores = box[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    centre_x = box[0]
                    centre_y = box[1]
                    w = box[2]
                    h = box[3]
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([round(centre_x),round(centre_y),round(w),round(h)])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5) # 마지막 숫자가 NMS 관여
        for i in indices:
            bbox = {}
            box = boxes[i]
            centerX = int(box[0] * originalxpcnt)
            if centerX < 0:
                centerX = 0
            centerY = int(box[1] * originalypcnt)
            if centerY < 0:
                centerY = 0
            width = int(box[2] * originalxpcnt)
            if width < 0:
                width = 0
            height = int(box[3] * originalypcnt)
            if height < 0:
                height = 0
            classname = classes[class_ids[i]]
            bbox["label"] = classname
            bbox["coordinates"] = {"x": round(centerX), "y" :round(centerY), "width" : round(width), "height" : round(height)}
            annotations.append(bbox)
        jsonoutput["annotations"] = annotations
        jsonoutput = [jsonoutput]
        with open(source[:-1]+jsonfilename, 'w') as outfile:
            json.dump(jsonoutput, outfile, indent =2)
        jsonoutput = {}

# jsondet("C:/Users/Administrator/Desktop/Detection/data/images/*", "C:/Users/Administrator/Desktop/Detection/nakdong.onnx", "C:/Users/Administrator/Desktop/Detection/data/classes/rubbish.txt")


def xmldet(source, onnx, class_path):
    net = cv2.dnn.readNet(onnx, config=None, framework=None)
    xmlfile = LabelFile()
    imgformat = ['jpg', 'png', 'psd', 'tiff', 'gif', 'jpeg']
    with open(class_path) as f:
        classes = f.read().split('\n')
    # rubbish.txt = ['Styrofoam_Box', 'Styrofoam_Buoy', 'Styrofoam_Piece', 'PET_Bottle', 'Glass', 'Metal', 'Plastic_Buoy', 'Plastic_Buoy_China', 'Plastic_ETC', 'Net', 'Rope'] 
    # ES.txt = ['Echi', 'Star'] 

    #### 이미지에서 나타나는 detection bbox 정보별로 shapes 리스트에 첨부하고 labelFile_custom.save_pascal_voc_format()에 해당 이미지 파일과 관련된 필수 parameter 입력후 실행
    for image in glob.glob(source):
        p = PurePath(image)
        if p.parts[-1].split('.')[1] not in imgformat:
            if p.parts[-1].split('.')[1] not in ['json', 'xml', 'txt']:
                print('Check the image file type.', p.parts[-1])
            continue
        img = cv2.imread(image)
        blob = cv2.dnn.blobFromImage(img, scalefactor= 1/255, size = (640, 640), swapRB=False)
        net.setInput(blob)
        prob = net.forward()
        filename = p.parts[-1].split('.')[0] +'.xml'
        shapes = []
        class_ids = []
        confidences = []
        boxes = []
        originalxpcnt = img.shape[1]/640 #원본에서 resize된 너비 비율
        originalypcnt = img.shape[0]/640 #원본에서 resize된 높이 비율
        for box in prob[0]:
            if box[4] > 0.5:
                scores = box[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    centre_x = box[0]
                    centre_y = box[1]
                    w = box[2]
                    h = box[3]
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([round(centre_x),round(centre_y),round(w),round(h)])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5) # 마지막 숫자가 NMS 관여
        for i in indices:
            shape = {}
            box = boxes[i]
            centerX = int(box[0] * originalxpcnt)
            if centerX < 0:
                centerX = 1
            centerY = int(box[1] * originalypcnt)
            if centerY < 0:
                centerY = 1
            width = int(box[2] * originalxpcnt)
            if width < 0:
                width = 1
            height = int(box[3] * originalypcnt)
            if height < 0:
                height = 1
            classname = classes[class_ids[i]]
            shape["label"] = classname
            shape["points"] = [round(centerX - width/2), round(centerY - height/2), round(centerX + width/2), round(centerY + height/2)]
            shape["difficult"] = False
            shapes.append(shape)
        xmlfile.save_pascal_voc_format(filename, shapes, str(p))

# xmldet("C:/Users/Administrator/Desktop/Detection/data/images/*", "C:/Users/Administrator/Desktop/Detection/nakdong.onnx", "C:/Users/Administrator/Desktop/Detection/data/classes/rubbish.txt")



def yolodet(source, onnx, class_path):
    net = cv2.dnn.readNet(onnx, config=None, framework=None)
    yolofile = LabelFile()
    imgformat = ['jpg', 'png', 'psd', 'tiff', 'gif', 'jpeg']
    with open(class_path) as f:
        classes = f.read().split('\n')
    # rubbish.txt = ['Styrofoam_Box', 'Styrofoam_Buoy', 'Styrofoam_Piece', 'PET_Bottle', 'Glass', 'Metal', 'Plastic_Buoy', 'Plastic_Buoy_China', 'Plastic_ETC', 'Net', 'Rope'] 
    # ES.txt = ['Echi', 'Star'] 

    #### 이미지에서 나타나는 detection bbox 정보별로 shapes 리스트에 첨부하고 labelFile_custom.save_yolo_format()에 해당 이미지 파일과 관련된 필수 parameter 입력후 실행
    for image in glob.glob(source):
        p = PurePath(image)
        if p.parts[-1].split('.')[1] not in imgformat:
            if p.parts[-1].split('.')[1] not in ['json', 'xml', 'txt']:
                print('Check the image file type.', p.parts[-1])
            continue
        img = cv2.imread(image)
        blob = cv2.dnn.blobFromImage(img, scalefactor= 1/255, size = (640, 640), swapRB=False)
        net.setInput(blob)
        prob = net.forward()
        filename = p.parts[-1].split('.')[0] +'.txt'
        shapes = []
        class_ids = []
        confidences = []
        boxes = []
        originalxpcnt = img.shape[1]/640 #원본에서 resize된 너비 비율
        originalypcnt = img.shape[0]/640 #원본에서 resize된 높이 비율
        for box in prob[0]:
            if box[4] > 0.5:
                scores = box[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    centre_x = box[0]
                    centre_y = box[1]
                    w = box[2]
                    h = box[3]
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([round(centre_x),round(centre_y),round(w),round(h)])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5) # 마지막 숫자가 NMS 관여
        for i in indices:
            shape = {}
            box = boxes[i]
            centerX = int(box[0] * originalxpcnt)
            if centerX < 0:
                centerX = 1
            centerY = int(box[1] * originalypcnt)
            if centerY < 0:
                centerY = 1
            width = int(box[2] * originalxpcnt)
            if width < 0:
                width = 1
            height = int(box[3] * originalypcnt)
            if height < 0:
                height = 1
            classname = classes[class_ids[i]]
            shape["label"] = classname
            shape["points"] = [round(centerX - width/2), round(centerY - height/2), round(centerX + width/2), round(centerY + height/2)]
            shape["difficult"] = False
            shapes.append(shape)
        yolofile.save_yolo_format(filename, shapes, str(p), classes, class_path)

# yolodet("C:/Users/Administrator/Desktop/Detection/data/images/*", "C:/Users/Administrator/Desktop/Detection/nakdong.onnx", "C:/Users/Administrator/Desktop/Detection/data/classes/rubbish.txt")
