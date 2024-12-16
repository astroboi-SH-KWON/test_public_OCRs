from flask import Flask, request, render_template, abort, make_response
import publicOCRsProcessor
from config import config
import utils
import time
import json
import base64
import uuid
import os
import cv2


app = Flask('public OCRs')
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=['GET', 'POST'])
def publicOCRs_api():
    st_time = time.perf_counter()
    data = request.data.decode('utf-8')
    data = json.loads(data)
    ocr_obj = data['ocr_object']
    ocr_mdl = data['ocr_model']
    if 'gpu_flag' in data:
        gpu_flag = data['gpu_flag']  # (String) 'True', 'False'
    else:
        gpu_flag = 'False'

    if utils.is_hexadecimal(ocr_obj):
        print("is_hexadecimal >>>>>>>>>>>>>>>>>>>>>>>>")
        ocr_obj_bytes = bytes.fromhex(ocr_obj)
        ocr_obj_img = utils.bts_to_img(ocr_obj_bytes)
    elif utils.is_base64(ocr_obj):
        print("is_Base64 >>>>>>>>>>>>>>>>>>>>>>>>")
        ocr_obj_bytes = base64.b64decode(ocr_obj)
        ocr_obj_img = utils.bts_to_img(ocr_obj_bytes)
    else:
        print("[ERROR-onyOCR_api] Choose decoding options")
        raise Exception

    try:
        now = time.time()
        fl_nm = str(uuid.uuid4())
        os.makedirs("./images/tmp_ocr", exist_ok=True)
        utils.remove_files_by_days("./images/tmp_ocr", now)
        cv2.imwrite(f"./images/tmp_ocr/ocr_img_{fl_nm}.jpg", ocr_obj_img)

        if ocr_mdl == "tesseract" or ocr_mdl == "tesseract_ony" or ocr_mdl == "tesseract_pub":
            ocr_obj_img = utils.load_img_by_PIL(f"./images/tmp_ocr/ocr_img_{fl_nm}.jpg")
        elif ocr_mdl == "easyocr" or ocr_mdl == "easyocr_cropped":
            ocr_obj_img = utils.load_img_by_cv2(f"./images/tmp_ocr/ocr_img_{fl_nm}.jpg")
        else:
            raise ValueError("Check OCR model.")
    except Exception as err:
        print(err)
        return abort(make_response(str(err), 500))
    print(f"load_img_by_cv2 ::: {time.perf_counter() - st_time} sec")

    ocr_st_time = time.perf_counter()
    public_ocr = publicOCRsProcessor.publicOCRs(gpu_flag)
    if ocr_mdl == "tesseract" or ocr_mdl == "tesseract_ony" or ocr_mdl == "tesseract_pub":
        result_text = public_ocr.get_ocr_result(ocr_obj_img, tessdata_prefix=config.TESSDATA_PREFIX[ocr_mdl])
    elif ocr_mdl == "easyocr" or ocr_mdl == "easyocr_cropped":
        result_text = public_ocr.get_ocr_result(ocr_obj_img, ocr_mdl)
    else:
        raise ValueError("Check OCR model. [public_OCRs_api.py]")

    print(f"publicOCRs_api {ocr_mdl} ::: {time.perf_counter() - ocr_st_time} sec")
    # # make_response for not only English but also non-English like Korean
    resp = make_response(json.dumps(result_text, ensure_ascii=False).encode("utf-8"))
    return resp


@app.route("/test_publicOCRs_api", methods=['GET', 'POST'])
def test_publicOCRs_api():
    st_time = time.perf_counter()
    data = request.data.decode('utf-8')
    data = json.loads(data)
    ocr_obj = data['ocr_object']
    ocr_mdl = data['ocr_model']
    if 'gpu_flag' in data:
        gpu_flag = data['gpu_flag']  # (String) 'True', 'False'
    else:
        gpu_flag = 'False'

    if utils.is_hexadecimal(ocr_obj):
        print("is_hexadecimal >>>>>>>>>>>>>>>>>>>>>>>>")
        ocr_obj_bytes = bytes.fromhex(ocr_obj)
        ocr_obj_img = utils.bts_to_img(ocr_obj_bytes)
    elif utils.is_base64(ocr_obj):
        print("is_Base64 >>>>>>>>>>>>>>>>>>>>>>>>")
        ocr_obj_bytes = base64.b64decode(ocr_obj)
        ocr_obj_img = utils.bts_to_img(ocr_obj_bytes)
    else:
        print("[ERROR-onyOCR_api] Choose decoding options")
        raise Exception

    try:
        now = time.time()
        fl_nm = str(uuid.uuid4())
        os.makedirs("./images/tmp_ocr", exist_ok=True)
        utils.remove_files_by_days("./images/tmp_ocr", now)
        cv2.imwrite(f"./images/tmp_ocr/ocr_img_{fl_nm}.jpg", ocr_obj_img)

        if ocr_mdl == "tesseract" or ocr_mdl == "tesseract_ony" or ocr_mdl == "tesseract_pub":
            ocr_obj_img = utils.load_img_by_PIL(f"./images/tmp_ocr/ocr_img_{fl_nm}.jpg")
        elif ocr_mdl == "easyocr" or ocr_mdl == "easyocr_cropped":
            ocr_obj_img = utils.load_img_by_cv2(f"./images/tmp_ocr/ocr_img_{fl_nm}.jpg")
        else:
            raise ValueError("Check OCR model.")
    except Exception as err:
        print(err)
        return abort(make_response(str(err), 500))
    print(f"load_img_by_cv2 ::: {time.perf_counter() - st_time} sec")

    ocr_st_time = time.perf_counter()

    public_ocr = publicOCRsProcessor.publicOCRs(gpu_flag)
    if ocr_mdl == "tesseract" or ocr_mdl == "tesseract_ony" or ocr_mdl == "tesseract_pub":
        result_text = public_ocr.get_ocr_result(ocr_obj_img, tessdata_prefix=config.TESSDATA_PREFIX[ocr_mdl])
    elif ocr_mdl == "easyocr" or ocr_mdl == "easyocr_cropped":
        result_text = public_ocr.get_ocr_result(ocr_obj_img, ocr_mdl)
    else:
        raise ValueError("Check OCR model. [public_OCRs_api.py]")

    working_time = time.perf_counter() - ocr_st_time
    print(f"publicOCRs_api {ocr_mdl} ::: {working_time} sec\nresult_text [{result_text}]\n\n")
    # # make_response for not only English but also non-English like Korean
    resp = make_response(json.dumps({'result_text': result_text, 'working_time': working_time}, ensure_ascii=False).encode("utf-8"))
    return resp


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True, host=config.HOME_URL, port=config.HOME_PORT)

"""
sudo lsof -i :8026  
kill -9 {PID}
"""