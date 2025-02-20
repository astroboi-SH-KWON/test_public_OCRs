import pytesseract
import easyocr
from easyocr.utils import reformat_input
from paddleocr import PaddleOCR
import os
import imutils


class publicOCRs:
    """
    https://www.google.com/search?q=%ED%8C%8C%EC%9D%B4%EC%8D%AC+tesseract+%ED%95%99%EC%8A%B5&rlz=1C5CHFA_enKR1061KR1061&oq=&gs_lcrp=EgZjaHJvbWUqCQgBECMYJxjqAjIJCAAQIxgnGOoCMgkIARAjGCcY6gIyCQgCECMYJxjqAjIJCAMQIxgnGOoCMgkIBBAjGCcY6gIyCQgFECMYJxjqAjIJCAYQIxgnGOoCMgYIBxBFGEDSAQoyMzMxNjZqMGo3qAIHsAIB&sourceid=chrome&ie=UTF-8
    easyocr : https://velog.io/@hyunk-go/%ED%81%AC%EB%A1%A4%EB%A7%81-Tesseract-OCR-EasyOCR-OpenCV-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%ED%95%99%EC%8A%B5
    이미지 전처리 : https://yunwoong.tistory.com/76
    https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv
    https://github.com/yardstick17/image_text_reader

    """
    def __init__(self, gpu_flag, pixel_threshold=60, text_threshold=0.5):
        self.gpu_flag = False
        if gpu_flag.lower() == 'true':
            self.gpu_flag = True
        self.img_pre = ImagePreprocessor()
        self.pixel_threshold = pixel_threshold
        self.text_threshold = text_threshold

    def get_ocr_result(self, img, which_ocr='tesseract', tessdata_prefix=''):
        """
        Get public OCRs results
        :param img:
        :param which_ocr: (String) OCR model
        :param tessdata_prefix: (String) /tessdata path
        :return: OCR results
        """
        img = imutils.resize(img, height=max(self.pixel_threshold, img.shape[0]))
        if which_ocr == 'tesseract':
            os.environ['TESSDATA_PREFIX'] = tessdata_prefix
            *_, img_cv_grey = self.img_pre.preprocess_image(img)
            return self.get_tesseract_result(img_cv_grey)
        elif which_ocr == 'easyocr':
            return self.get_easyocr_result(img)
        elif which_ocr == 'easyocr_cropped':
            return self.get_easyocr_result_from_cropped(img)
        elif which_ocr == 'paddleocr':
            img, *_ = self.img_pre.preprocess_image(img)
            return self.get_paddleocr_result(img)
        else:
            raise ValueError("Check OCR model.")

    def get_tesseract_result(self, img):
        """
        Get tesseract OCR results
        :param img: (PIL) image
        :return: (String) OCR results
        """
        return pytesseract.image_to_string(img, lang='kor+eng')

    def get_easyocr_result(self, img):
        """
        Get easyOCR results with long text image
        :param img: image object(openCV2), image path, image url and etc
        :return: (List) OCR results
        """
        reader = easyocr.Reader(['ko', 'en'], gpu=self.gpu_flag)
        return reader.readtext(img, detail=0)

    def get_easyocr_result_from_cropped(self, img):
        """
        Get easyOCR results with cropped image without text detection
        :param img: image object(openCV2), image path, image url and etc
        :return: (List) OCR results
        """
        reader = easyocr.Reader(['ko', 'en'], gpu=self.gpu_flag)
        return reader.recognize(img, reformat=True, detail=0)

    def get_paddleocr_result(self, img):
        """
        Get Paddle OCR results
        :param img:
        :return:
        """
        paddle_ocr = PaddleOCR(
            lang='korean',  # other lang also available 'korean', 'en'
            use_angle_cls=False,
            use_gpu=self.gpu_flag,  # using cuda will conflict with pytorch in the same process
            show_log=False,
            max_batch_size=1024,
            use_dilation=True,  # improves accuracy
            det_db_score_mode='slow',  # improves accuracy
            rec_batch_num=1024)
        result = paddle_ocr.ocr(img, cls=False)[0]
        print(f"{type(result)} \nresult:{result}")
        if result is None:
            return [""]
        # coord = [item[0] for item in result if item[1][1] > self.text_threshold]
        return [item[1][0] for item in result if item[1][1] > self.text_threshold]


IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


class ImagePreprocessor:

    def __init__(self, pixel_threshold=60):
        self.pixel_threshold = pixel_threshold

    def preprocess_image(self, img):
        """
        load and preprocess image to (openCV2) color image for detection, (openCV2) grey image for OCR
        :param img: image object(openCV2), image path, image url and etc
        :return: img_cv_color, img_cv_grey
        """
        img_cv_color, img_cv_grey = reformat_input(img)
        return img_cv_color, img_cv_grey
