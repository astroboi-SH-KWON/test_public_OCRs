from config import config
from utils import *
import requests
import base64
import time
import glob


def get_ocr_result_by_API(api_url, img_path, gpu, which_ocr='pytesseract', is_hexadecimal=True):
    if is_hexadecimal:
        print(f"\nis_hexadecimal {'|' * 30}\n")
        ocr_obj = image_to_bts(img_path)

        st_time = time.perf_counter()
        response = requests.post(api_url, json={
            'ocr_object': ocr_obj.hex(),
            'ocr_model': which_ocr.lower(),
            'gpu_flag': str(gpu),
        })
    else:
        print(f"\nis_Base64 {'|' * 30}\n")
        ocr_obj = open(img_path, 'rb')

        st_time = time.perf_counter()
        response = requests.post(api_url, json={
            'ocr_object': base64.b64encode(ocr_obj.read()).decode(),
            'ocr_model': which_ocr.lower(),
            'gpu_flag': str(gpu),
        })

        ocr_obj.close()

    ocr_result = response.json()
    print(f"test_{which_ocr}_api :::::: {time.perf_counter() - st_time} sec\n")
    return ocr_result


if __name__ == "__main__":
    # url = "http://" + str(config.HOME_URL) + ":" + str(config.HOME_PORT) + "/test_publicOCRs_api"
    url = "http://192.168.3.138:" + str(config.HOME_PORT) + "/test_publicOCRs_api"
    gpu = False
    if '192.168.3.138' in url:
        gpu = True

    img_path_list = glob.glob("./tests/data/*.jpg")
    for img_path in img_path_list:
        print(img_path.split("/")[-1])
        which_ocr = 'tesseract_ony'  # tesseract_ony, tesseract_pub, tesseract
        print(f"{which_ocr}>>>")
        print(f"[{get_ocr_result_by_API(url, img_path, gpu, which_ocr=which_ocr, is_hexadecimal=True)}]")

        which_ocr = 'easyocr'
        print(f"{which_ocr}>>>")
        print(get_ocr_result_by_API(url, img_path, gpu, which_ocr=which_ocr, is_hexadecimal=True))
        print(f"\n\n")