import time
import os
import platform
import sys
import string
import glob
import re
import copy
from shutil import move as shutil_move
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2
import time
import os

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.models import load_model
from keras import backend as K

if K.backend() != 'tensorflow':
    raise RuntimeError(f"Please set your keras.json to use TensorFlow. It is currently using {keras.backend.backend()}")

position_model = tf.lite.Interpreter(model_path="models/ColorNet/mlp_proposal.tflite")
position_model.allocate_tensors()
position_input_details = position_model.get_input_details()
position_output_details = position_model.get_output_details()

K_position_model = load_model("models/ColorNet//mlp_proposal_k.hdf5")
position_function = K.function([K_position_model.layers[0].input,
                                K_position_model.layers[1].input,
                                K.learning_phase()],
                               [K_position_model.layers[-1].output])

discriminator_model = tf.lite.Interpreter(model_path="models/ColorNet/discriminator.tflite")
discriminator_model.allocate_tensors()
discriminator_input_details = discriminator_model.get_input_details()
discriminator_output_details = discriminator_model.get_output_details()


def ocv_to_pil(im):
    pil_image = np.array(im)
    pil_image = Image.fromarray(pil_image)
    return pil_image


def _position_with_uncertainty(x, n_iter=10):
    result = []

    for i in range(n_iter):
        result.append(position_function([x[0], x[1], 1]))

    result = np.array(result)
    uncertainty = result.var(axis=0)
    prediction = result.mean(axis=0)
    return prediction, uncertainty


def _legacy_regions(im, im_hsv, image_width, image_height, whole_extrema, stride_style='quick', stride=25,
                    partition_size=125, over_crop=0, hard_cut_value=50):
    possible_positions = []
    hists_rgb = []
    hists_hsv = []

    if stride_style == 'whole':
        for r in range(-over_crop, (image_height - partition_size) // stride + over_crop):
            for c in range(-over_crop, (image_width - partition_size) // stride + over_crop):
                x1, y1 = c * stride, r * stride
                x2, y2 = x1 + partition_size, y1 + partition_size
                partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
                partitioned_im_hsv = partitioned_im_hsv.resize((125, 125))
                extrema = partitioned_im_hsv.getextrema()
                extrema = extrema[1][1]
                if whole_extrema - hard_cut_value < extrema:
                    possible_positions.append((x1, y1, x2, y2))
                    partitioned_im = im.crop((x1, y1, x2, y2))
                    partitioned_im = partitioned_im.resize((125, 125))
                    hists_rgb.append(partitioned_im.histogram())
                    hists_hsv.append(partitioned_im_hsv.histogram())

    elif stride_style == 'quick':
        for c in range(-over_crop, (image_width - partition_size) // stride + over_crop):
            x1, y1 = c * stride, 0
            x2, y2 = x1 + partition_size, y1 + partition_size
            partitioned_im = im.crop((x1, y1, x2, y2))
            possible_positions.append((x1, y1, x2, y2))
            partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))

            hist = partitioned_im.histogram()
            hist_hsv = partitioned_im_hsv.histogram()

            hists_rgb.append(hist)
            hists_hsv.append(hist_hsv)

            x1, y1 = c * stride, image_height - partition_size
            x2, y2 = x1 + partition_size, y1 + partition_size
            partitioned_im = im.crop((x1, y1, x2, y2))
            possible_positions.append((x1, y1, x2, y2))
            partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
            hist = partitioned_im.histogram()
            hist_hsv = partitioned_im_hsv.histogram()

            hists_rgb.append(hist)
            hists_hsv.append(hist_hsv)

        for r in range(-over_crop, (image_height - partition_size) // stride + over_crop):
            x1, y1 = 0, r * stride
            x2, y2 = x1 + partition_size, y1 + partition_size
            partitioned_im = im.crop((x1, y1, x2, y2))
            possible_positions.append((x1, y1, x2, y2))
            partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
            hist = partitioned_im.histogram()
            hist_hsv = partitioned_im_hsv.histogram()

            hists_rgb.append(hist)
            hists_hsv.append(hist_hsv)

            x1, y1 = image_width - partition_size, r * stride
            x2, y2 = x1 + partition_size, y1 + partition_size
            partitioned_im = im.crop((x1, y1, x2, y2))
            possible_positions.append((x1, y1, x2, y2))
            partitioned_im_hsv = im_hsv.crop((x1, y1, x2, y2))
            hist = partitioned_im.histogram()
            hist_hsv = partitioned_im_hsv.histogram()

            hists_rgb.append(hist)
            hists_hsv.append(hist_hsv)
    elif stride_style == 'ultraquick':
        positions = [(0, 0, partition_size, partition_size),
                     (0, image_height - partition_size, partition_size, image_height),
                     (image_width - partition_size, 0, image_width, partition_size),
                     (image_width - partition_size, image_height - partition_size, image_width, image_height)]

        for position in positions:
            partitioned_im = im.crop(position)
            possible_positions.append(position)
            partitioned_im_hsv = im_hsv.crop(position)
            hists_rgb.append(partitioned_im.histogram())
            hists_hsv.append(partitioned_im_hsv.histogram())
    else:
        raise InvalidStride

    return hists_rgb, hists_hsv, possible_positions


def scale_images_with_info(im, largest_dim=1875):
    image_height, image_width = im.shape[0:2]

    if image_width > image_height:
        reduced_im = cv2.resize(im,
                                (largest_dim,
                                 round((largest_dim / image_width) * image_height)),
                                interpolation=cv2.INTER_NEAREST)
    else:
        reduced_im = cv2.resize(im,
                                (round((largest_dim / image_height) * image_width),
                                 largest_dim),
                                interpolation=cv2.INTER_NEAREST)
    return (image_width, image_height), reduced_im


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


def process_crc_small(im, original_size, stride_style='whole', stride=25, partition_size=125,
                      discriminator_floor=0.90, over_crop=1, hard_cut_value=50, high_precision=False,
                      full_tf=True):
    nim = im
    im = ocv_to_pil(im)
    im_hsv = im.convert("HSV")
    whole_extrema = im_hsv.getextrema()
    whole_extrema = whole_extrema[1][1]
    start = time.time()
    image_width, image_height = im.size
    original_width, original_height = original_size
    cv_image = cv2.cvtColor(nim, cv2.COLOR_RGB2HSV)
    partition_area = partition_size * partition_size
    contour_area_floor = partition_area // 10
    contour_area_ceiling = partition_area // 0.5
    squares = find_squares(cv_image,
                           contour_area_floor=contour_area_floor,
                           contour_area_ceiling=contour_area_ceiling,
                           leap=17)

    squares = np.array(squares)

    part_im = []
    possible_positions = []

    for square in squares:
        M = cv2.moments(square)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        location = (
            cX - (partition_size // 2), cY - (partition_size // 2), cX + (partition_size // 2),
            cY + (partition_size // 2))
        part_image = im.crop(location)
        part_image = part_image.resize((125, 125))
        extrema = part_image.convert("HSV").getextrema()
        extrema = extrema[1][1]
        if whole_extrema - hard_cut_value < extrema:
            part_im.append(part_image)
            possible_positions.append(location)

    if len(part_im) != 0:
        inference_type = 'find_squares'
        hists_rgb = []
        hists_hsv = []
        for im in part_im:
            im_hsv = im.convert("HSV")
            hists_rgb.append(im.histogram())
            hists_hsv.append(im_hsv.histogram())

    else:
        inference_type = 'legacy'
        hists_rgb, hists_hsv, possible_positions = _legacy_regions(im=im, im_hsv=im_hsv,
                                                                   image_width=image_width,
                                                                   image_height=image_height,
                                                                   whole_extrema=whole_extrema,
                                                                   stride_style=stride_style, stride=stride,
                                                                   partition_size=partition_size,
                                                                   over_crop=over_crop,
                                                                   hard_cut_value=hard_cut_value)

        hists_rgb = np.array(hists_rgb, dtype=np.uint16)
        hists_hsv = np.array(hists_hsv, dtype=np.uint16)

        position_predictions = []
        position_start = time.time()

    if full_tf:
        position_prediction, position_uncertainty = _position_with_uncertainty([hists_rgb, hists_hsv], 10)

        only_cc_position_uncertainty = position_uncertainty[0][:, 1]
        only_cc_position_prediction = position_prediction[0][:, 1]

        indices = [index for index in range(len(only_cc_position_prediction))]
        position_uncertainty, position_predictions, indices = \
            (list(t) for t in zip(*sorted(zip(only_cc_position_uncertainty, only_cc_position_prediction, indices))))

        max_pred = max(position_predictions)
        for _j, position_prediction in enumerate(position_predictions):
            try:
                if position_prediction < (max_pred - 0.001):
                    del position_prediction
                    del position_uncertainty[_j]
                    del indices[_j]
            except IndexError:
                break

    else:
        indices = [i for i in range(len(hists_rgb))]
        for i in indices:

            try:
                position_model.set_tensor(position_model.get_input_details()[0]['index'], [hists_rgb[i]])
                position_model.set_tensor(position_model.get_input_details()[1]['index'], [hists_hsv[i]])
                position_model.invoke()

                position_predictions.append(
                    position_model.get_tensor(position_model.get_output_details()[0]['index'])[0][0])
            except:
                position_predictions.append(np.array([[1, 0]], dtype=np.float32).tolist())

        position_predictions, indices = (list(t) for t in zip(*sorted(zip(position_predictions, indices))))
        position_predictions.reverse()
        indices.reverse()

    highest_prob_images = []
    highest_prob_positions = []

    if inference_type == 'find_squares':
        highest_prob_images = [np.array(part_im[k]) for k in indices]
        highest_prob_positions = [possible_positions[k] for k in indices]
    else:
        for i in indices:
            highest_prob_images.append(np.array(im.crop(possible_positions[i]).resize((125, 125))))
            highest_prob_positions.append(possible_positions[i])

    highest_prob_images_pred = np.array(highest_prob_images, dtype=np.float32) / 255

    if inference_type == 'find_squares':
        best_image = Image.fromarray(highest_prob_images[0])
        best_location = highest_prob_positions[0]
    else:
        for i, highest_prob_image in enumerate(highest_prob_images):
            discriminator_model.set_tensor(discriminator_model.get_input_details()[0]['index'],
                                           [highest_prob_images_pred[i]])
            discriminator_model.invoke()
            disc_value = discriminator_model.get_tensor(discriminator_model.get_output_details()[0]['index'])[0][1]
            if disc_value > discriminator_floor:
                best_image = Image.fromarray(highest_prob_image)
                best_location = highest_prob_positions[i]
                break
        else:
            raise DiscriminatorFailed

    best_image = np.array(best_image, dtype=np.uint8)
    x1, y1, x2, y2 = best_location[0], best_location[1], best_location[2], best_location[3]
    if high_precision:
        best_image, best_square = high_precision_cc_crop(best_image)
        ratio = 125 / partition_size
        best_square = [best_square[0] // ratio,
                       best_square[1] // ratio,
                       best_square[2] // ratio,
                       best_square[3] // ratio]
        x1, y1, x2, y2 = best_location[0] + best_square[0], best_location[1] + best_square[1], \
                         best_location[0] + best_square[2], best_location[1] + best_square[3]

    xc = np.array([x1, x2])
    yc = np.array([y1, y2])
    xc = np.clip(xc, 0, original_size[0])
    yc = np.clip(yc, 0, original_size[1])
    x1, y1, x2, y2 = xc[0], yc[0], xc[1], yc[1]

    prop_x1, prop_y1, prop_x2, prop_y2 = x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height

    scaled_x1, scaled_y1, scaled_x2, scaled_y2 = int(prop_x1 * original_width), \
                                                 int(prop_y1 * original_height), \
                                                 int(prop_x2 * original_width), \
                                                 int(prop_y2 * original_height)

    end = time.time()

    cc_crop_time = round(end - start, 3)
    print(f"Inference type: {inference_type}")
    return (scaled_x1, scaled_y1, scaled_x2, scaled_y2), best_image, cc_crop_time


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


def predict_color_chip_whitevals(cropped_cc, crc_type=""):
    def _remove_wrong_white_loc(_cropped_cc):
        prop_diff_height = abs(h / 2 - maxLoc[1]) / h
        prop_diff_width = abs(w / 2 - maxLoc[0]) / w

        if prop_diff_height > prop_diff_width and maxLoc[1] > h / 2:
            _cropped_cc = _cropped_cc[0:maxLoc[1] - 2, 0:w]
        elif prop_diff_height > prop_diff_width and maxLoc[1] < h / 2:
            _cropped_cc = _cropped_cc[maxLoc[1] + 2:h, 0:w]
        elif prop_diff_height < prop_diff_width and maxLoc[0] > w / 2:
            _cropped_cc = _cropped_cc[0:h, 0:maxLoc[0] - 2]
        else:
            _cropped_cc = _cropped_cc[0:h, maxLoc[0] + 2:w]

        return _cropped_cc

    for _ in range(50):
        grayImg = cv2.cvtColor(cropped_cc, cv2.COLOR_RGB2GRAY)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
        var_threshold = int((maxVal - minVal) * .1)
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
        contour_area_floor = area // 75
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
            if crc_type in ['Tiffen / Kodak Q-13  (8")',
                            ]:

                if 1.45 < ratio < 1.75:
                    break
            else:
                if 0.85 < ratio < 1.15:
                    break
        else:
            cropped_cc = _remove_wrong_white_loc(cropped_cc)
            continue
        break
    else:
        raise ValueError("Failed to find the white patch!")

    extracted = cropped_cc[mask != 0]
    extracted = extracted.reshape(-1, extracted.shape[-1])
    mode_white = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=extracted)

    return list(mode_white), minVal


def white_balance(img, avg_white):
    r, g, b = avg_white
    im = img.transpose(2, 0, 1)
    im = im.astype(np.int32)

    im[0] = np.minimum(im[0] * (255 / float(r) - 0.2), 255)
    im[1] = np.minimum(im[1] * (255 / float(g) - 0.2), 255)
    im[2] = np.minimum(im[2] * (255 / float(b) - 0.2), 255)
    img = im.transpose(1, 2, 0).astype(np.uint8)
    return img

