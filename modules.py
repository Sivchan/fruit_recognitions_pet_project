import copy
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
from skimage import transform

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def prepare_input(img, input_height, input_width):
    img = cv2.imread(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    aspect = img.shape[1] / float(img.shape[0])
    if (aspect > 1):
        res = int(aspect * input_height)
        img_res = transform.resize(img_rgb, (input_width, res))
    if (aspect < 1):
        res = int(input_width / aspect)
        img_res = transform.resize(img_rgb, (res, input_height))
    if (aspect == 1):
        img_res = transform.resize(img_rgb, (input_width, input_height))
    img_res /= 255.0
    img_res = torchvision.transforms.ToTensor()(img_res)
    return img_res


def plot_img_bbox(img, target, num_detection, treshholds=None, inst_classes=[]):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    dir_jpg = 'static/Results/' + str(num_detection) + '_picture_with_det' + '.jpg'
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    a.imshow(img, cmap='gray')
    for number, box in enumerate(target['boxes']):
        if target['scores'][number] > treshholds:
            x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                    width, height,
                                    linewidth=2,
                                    edgecolor='r',
                                    facecolor='none')
            a.add_patch(rect)
            a.text(x, y,
                   (round(target['scores'][number].item() * 100, 2), inst_classes[target['labels'][number].item()]),
                   bbox=dict(facecolor='white', alpha=0.5))

    fig.savefig(dir_jpg)


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = copy.deepcopy(orig_prediction)
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def torch_to_pil(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')


def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def init_model():
    num_classes = 4

    model = get_object_detection_model(num_classes)
    weight = 'static/Others/pretrained_weights.pth'
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))

    return model


def download_img(url, num_detection):
    dir_img = 'static/Results/' + str(num_detection) + '_picture' + '.jpg'
    img = urllib.request.urlopen(url).read()
    with open(dir_img, "wb") as file:
        file.write(img)
    return dir_img


def make_detection(img, num_detection):
    fruit_classes = ['background', 'apple', 'banana', 'orange']

    model = init_model()
    model.eval()
    with torch.no_grad():
        prediction = model([img])[0]

    nms_prediction = apply_nms(prediction, iou_thresh=0.3)
    plot_img_bbox(torch_to_pil(img), nms_prediction, num_detection, treshholds=0.5, inst_classes=fruit_classes)


def make_result(url, num_detection):
    img = download_img(url, num_detection)
    img = prepare_input(img, 480, 480)
    make_detection(img, num_detection)