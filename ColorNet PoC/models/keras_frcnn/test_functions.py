from __future__ import division
import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import sys
import pickle
import time
from models.keras_frcnn import config
from models.keras_frcnn.test_functions import *
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from models.keras_frcnn import roi_helpers
from models.keras_frcnn.config import Config
from timeit import default_timer as Timer
import models.keras_frcnn.vgg as nn

C = Config()
sys.setrecursionlimit(40000)

class_mapping = {'0': 0, '1': 1, 'bg': 2}

class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
num_features = 128

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn.nn_base(img_input, trainable=True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

print(C.model_path)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)
model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []
classes = {}
bbox_threshold = 0.8

visualise = True
img_path = "sample_images/big/"


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img, contour_area_floor=850, contour_area_ceiling=100000, leap=6, k_size=5):
    img = cv2.GaussianBlur(img, (k_size, k_size), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, leap):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=3)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                contourArea = cv2.contourArea(cnt)
                if len(cnt) == 4 and contour_area_floor < contourArea < contour_area_ceiling \
                        and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                                      for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def format_img_size(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def white_balance(img, avg_white):
    r, g, b = avg_white
    im = img.transpose(2, 0, 1)
    im = im.astype(np.int32)

    im[0] = np.minimum(im[0] * (255 / float(r) - 0.18), 255)
    im[1] = np.minimum(im[1] * (255 / float(g) - 0.18), 255)
    im[2] = np.minimum(im[2] * (255 / float(b) - 0.18), 255)
    img = im.transpose(1, 2, 0).astype(np.uint8)
    return img


def process_image_frcnn(img, key='0', pp_fix=0):
    X, ratio = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    [Y1, Y2, F] = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    try:
        bbox = np.array(bboxes[key])
    except KeyError:
        return 0, 0, 0, 0

    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=1)
    best_box_idx = 0
    highest = 0
    for idx, jk in enumerate(range(new_boxes.shape[0])):
        if jk > highest:
            highest = jk
            best_box_idx = idx
    (x1, y1, x2, y2) = new_boxes[best_box_idx, :]

    if pp_fix == 1:
        h, w, _ = img.shape
        if w > h:
            half_line = ratio * w / 2
            centroid = (x1 + x2) / 2
            if centroid > half_line:
                x1 = x1 + 80
                x2 = x2 + 140
            else:
                x1 = x1 - 140
                x2 = x2 - 80
        else:
            half_line = ratio * h / 2
            centroid = (y1 + y2) / 2

            if centroid > half_line:
                y1 = y1 + 80
                y2 = y2 + 140
            else:
                y1 = y1 - 140
                y2 = y2 - 80

    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
    return real_x1, real_y1, real_x2, real_y2


def predict_color_chip_whitevals(cropped_cc):
    def _remove_wrong_white_loc(cropped_cc):
        prop_diff_height = abs(h / 2 - maxLoc[1]) / h
        prop_diff_width = abs(w / 2 - maxLoc[0]) / w

        if prop_diff_height > prop_diff_width and maxLoc[1] > h / 2:
            cropped_cc = cropped_cc[0:maxLoc[1] - 2, 0:w]
        elif prop_diff_height > prop_diff_width and maxLoc[1] < h / 2:
            cropped_cc = cropped_cc[maxLoc[1] + 2:h, 0:w]
        elif prop_diff_height < prop_diff_width and maxLoc[0] > w / 2:
            cropped_cc = cropped_cc[0:h, 0:maxLoc[0] - 2]
        else:
            cropped_cc = cropped_cc[0:h, maxLoc[0] + 2:w]

        return cropped_cc

    for _ in range(10):
        grayImg = cv2.cvtColor(cropped_cc, cv2.COLOR_RGB2GRAY)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
        var_threshold = int((maxVal - minVal) * .5)
        cropped_cc = cropped_cc
        h, w, chn = cropped_cc.shape

        seed = maxLoc
        mask = np.zeros((h + 2, w + 2), np.uint8)
        floodflags = 8
        floodflags |= cv2.FLOODFILL_FIXED_RANGE
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        floodflags |= (int(maxVal) << 8)
        num, cropped_cc, mask, rect = cv2.floodFill(cropped_cc, mask, seed,
                                                    0,
                                                    (var_threshold,) * 3,
                                                    (var_threshold,) * 3,
                                                    floodflags)
        mask = mask[1:-1, 1:-1, ...]
        area = h * w
        contour_area_floor = area // 50
        contour_area_ceiling = area // 1
        squares = find_squares(mask,
                               contour_area_floor=contour_area_floor,
                               contour_area_ceiling=contour_area_ceiling)

        if len(squares) == 0:
            cropped_cc = _remove_wrong_white_loc(cropped_cc)
            continue

        squares = sorted(squares, key=cv2.contourArea, reverse=True)
        for square in squares:
            x_arr = square[..., 0]
            y_arr = square[..., 1]
            x1, y1, x2, y2 = np.min(x_arr), np.min(y_arr), np.max(x_arr), np.max(y_arr)
            square_width, square_height = x2 - x1, y2 - y1
            longest_side = max(square_width, square_height)
            shortest_side = min(square_width, square_height)
            ratio = longest_side / shortest_side

            if 0.85 < ratio < 1.15 or 1.45 < ratio < 1.75:
                break
        else:
            cropped_cc = _remove_wrong_white_loc(cropped_cc)
            continue
        break
    else:
        raise ValueError("Could not find the proper white square!")

    extracted = cropped_cc[mask != 0]
    extracted = extracted.reshape(-1, extracted.shape[-1])
    mode_white = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=extracted)

    return list(mode_white), minVal


def predict_color_chip_quadrant(original_size, scaled_crop_location):
    original_width, original_height = original_size
    x1, y1, x2, y2 = scaled_crop_location
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_width = original_width / 2
    half_height = original_height / 2

    if cx > half_width and cy < half_height:
        return 1
    elif cx < half_width and cy < half_height:
        return 2
    elif cx < half_width and cy > half_height:
        return 3
    elif cx > half_width and cy > half_height:
        return 4
    else:
        return None
