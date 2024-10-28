import numpy as np
import cv2
import string
import base64
import os
from PIL import Image
import socket


def bts_to_img(bts):
    """
    bytes array to cv2 image object
    :param bts: results from image_to_bts
    :return: cv2 image object
    """
    buff = np.frombuffer(bts, np.uint8)
    buff = buff.reshape(1, -1)
    img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return img


def image_to_bts(img_path):
    """
    Get image bytes from image
    :param img_path:  file path
    :return: Image bytes, WxHx3 ndarray
    """
    frame = cv2.imread(img_path)
    _, bts = cv2.imencode('.jpg', frame)
    bts = bts.tobytes()
    return bts


def is_hexadecimal(s):
    """
    Check if a string is hexadecimal or not
    :param s: String
    :return: True/False
    """
    return all(c in string.hexdigits for c in s)


def is_base64(s):
    """
    Check if a string is encoded by base64 or not
    :param s: String
    :return: True/False
    """
    try:
        if isinstance(s, str):
            # If there's any unicode here, an exception will be thrown and the function will return false
            s_bytes = bytes(s, 'ascii')
        elif isinstance(s, bytes):
            s_bytes = s
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(s_bytes)) == s_bytes
    except Exception as err:
        print(f"[ERROR-is_base64] {err}")
        return False


def remove_files_by_days(path, now, days=1):
    """
    Remove old files by days
    :param path: directory path
    :param now: time.time()
    :param days:
    """
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if os.stat(f).st_mtime < now - days * 86400:
            if os.path.isfile(f):
                os.remove(f)


def load_img_by_cv2(img_path, option=cv2.IMREAD_COLOR):
    """
    Load image by openCV2
    :param img_path:
    :param option: cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED (including alpha channel)
    :return: type openCV2
    """
    return cv2.imread(img_path, option)


def load_img_by_PIL(img_path):
    """
    Load image by PIL.Image
    :param img_path: String
    :return: type PIL.Image
    """
    return Image.open(img_path)


def get_ROI_xywh(test_img_path, window_name="get_ROI_xywh"):
    """
    Get Region of Interest
    :param test_img_path:
    :param window_name:
    :return:
    """
    cv2.namedWindow(window_name)
    test_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    cv2.imshow(window_name, test_img)

    return cv2.selectROI(window_name, test_img)


def get_local_ip_add():
    """
    Get ip address of local machine
    :return: (String) ip address
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        ip_add = s.getsockname()[0]
    return ip_add
