import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import math
import numpy as np
import pandas as pd
import cv2
import requests
import json
import socket


def current_time():
    """
    Gives current time
    :return: (String) Current time formated %Y-%m-%d %H:%M:%S
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def dict2df(my_dict, path):
    """
    Save python dictionary as csv using pandas dataframe
    :param my_dict: Dictionary like {"epoch": [1, 2], "accuracy": [0.5, 0.9]}
    :param path: /path/to/file.csv
    :return: (void) Save csv on specified path
    """
    df = pd.DataFrame.from_dict(my_dict, orient="columns")
    df.index.names = ['epoch']
    df.to_csv(path, index=True)


def convert_multiclass_mask(mask):
    """
    Transform multiclass mask [batch, num_classes, h, w] to [batch, h, w]
    :param mask: Mask to transform
    :return: Transformed multiclass mask
    """
    return mask.max(1)[1]


def reshape_masks(ndarray, to_shape, mask_reshape_method):
    """

    Args:
        ndarray: (np.array) Mask Array to reshape
        to_shape: (tuple) Final desired shape
        mask_reshape_method:

    Returns: (np.array) Reshaped array to desired shape

    """

    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape

    if mask_reshape_method == "padd":

        if h_in > h_out:  # center crop along h dimension
            h_offset = math.ceil((h_in - h_out) / 2)
            ndarray = ndarray[h_offset:(h_offset + h_out), :]
        else:  # zero pad along h dimension
            pad_h = (h_out - h_in)
            rem = pad_h % 2
            pad_dim_h = (math.ceil(pad_h / 2), math.ceil(pad_h / 2 + rem))
            # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
            npad = (pad_dim_h, (0, 0))
            ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

        if w_in > w_out:  # center crop along w dimension
            w_offset = math.ceil((w_in - w_out) / 2)
            ndarray = ndarray[:, w_offset:(w_offset + w_out)]
        else:  # zero pad along w dimension
            pad_w = (w_out - w_in)
            rem = pad_w % 2
            pad_dim_w = (math.ceil(pad_w / 2), math.ceil(pad_w / 2 + rem))
            npad = ((0, 0), pad_dim_w)
            ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    elif mask_reshape_method == "resize":
        ndarray = cv2.resize(ndarray.astype('float32'), (w_out, h_out))
    else:
        assert False, f"Unknown mask resize method '{mask_reshape_method}'"

    return ndarray  # reshaped


def reshape_volume(volume, to_shape, mask_reshape_method):
    """
    volume: (np.array) Volume Mask Array to reshape (slices, height, width)
    """
    res = []
    for c_slice in volume:
        res.append(reshape_masks(c_slice, to_shape, mask_reshape_method))
    return np.array(res)


def plot_save_pred(original_img, original_mask, pred_mask, save_dir, img_id):
    import warnings
    warnings.filterwarnings('ignore')

    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax1.imshow(original_img, cmap="gray")
    ax1.set_title("Original Image")

    masked = np.ma.masked_where(original_mask == 0, original_mask)
    ax2.imshow(original_img, cmap="gray")
    ax2.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax2.set_title("Original Overlay")

    masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    ax3.imshow(original_img, cmap="gray")
    ax3.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax3.set_title("Prediction Overlay")

    pred_filename = os.path.join(
        save_dir,
        f"mask_pred_{img_id}.png",
    )
    plt.savefig(pred_filename, dpi=200, pad_inches=0.2, bbox_inches='tight')
    plt.close()


def slack_message(message, channel, blocks=None):
    # https://keestalkstech.com/2019/10/simple-python-code-to-send-message-to-slack-channel-without-packages/
    if os.environ.get('SLACK_TOKEN') is not None:
        token = os.environ.get('SLACK_TOKEN')
    else:
        assert False, "Please set the environment variable SLACK_TOKEN if you want Slack notifications."
    if channel[0] != "#":
        channel = f"#{channel}"
    message = "[{}] {}".format(socket.gethostname().upper(), message)
    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': token,
        'channel': channel,
        'text': message,
        # https://slackmojis.com/
        'icon_url': "https://emojis.slackmojis.com/emojis/images/1453406830/264/success-kid.png?1453406830",
        'username': "Experiments Bot",
        'blocks': json.dumps(blocks) if blocks else None
    }).json()
