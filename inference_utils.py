import torch

import cv2
import numpy as np
import craft_utils
import imgproc
import matplotlib.pyplot as plt

from craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_detector(checkpoint_path, device):
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(checkpoint_path, map_location=device)))
    net.to(device)
    net.eval();
    return net


def test_net(net, image, image_size, mag_ratio, text_threshold, link_threshold, low_text, device, poly):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, image_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def visualize_detection(img, boxes, texts=None):
    img = np.array(img)[:,:,::-1].copy()

    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        if texts is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(img, "{}".format(texts[i]), (poly[0][0] + 1, poly[0][1] + 1), font, font_scale, (0, 0, 0),
                        thickness=1)
            cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

    plt.figure(figsize=(18,10))
    plt.imshow(img[:,:,::-1])
    plt.show()


