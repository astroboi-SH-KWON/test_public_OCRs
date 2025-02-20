import pytesseract
import easyocr
from easyocr.utils import reformat_input
from paddleocr import PaddleOCR
import os
import imutils
from PIL import Image
import cv2
import numpy as np
import tempfile


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
        if which_ocr == 'tesseract':
            os.environ['TESSDATA_PREFIX'] = tessdata_prefix
            # print(f"img.shape: {img.shape}")
            # resized = imutils.resize(img, height=max(self.pixel_threshold, img.shape[0]))
            # # resized = img.resize((int(max(self.pixel_threshold, img.size[1]) * img.size[0] / img.size[1]), max(self.pixel_threshold, img.size[1])))
            # print(f"resized.shape: {resized.shape}")
            # # *_, img_cv_grey = self.img_pre.preprocess_image(resized)

            img_cv_grey = self.img_pre.simple_preprocess_image(img)
            self.img_pre.invert_if_text_is_brighter(img_cv_grey)
            return self.get_tesseract_result(img_cv_grey)
        elif which_ocr == 'easyocr':
            return self.get_easyocr_result(img)
        elif which_ocr == 'easyocr_cropped':
            return self.get_easyocr_result_from_cropped(img)
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

    def simple_preprocess_image(self, img):
        resized = imutils.resize(img, height=max(self.pixel_threshold, img.shape[0]))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def invert_if_text_is_brighter(self, img_cv_grey, threshold_percentage=0.5):
        """
        Invert if text is bright than background
        :param img_cv_grey:
        :param threshold_percentage: 글자가 밝다고 판단하는 밝은 픽셀 비율의 임계값 (0~1 사이의 값)
        :return: (NumPy array)
        """
        # 평균 밝기 계산
        average_brightness = np.mean(img_cv_grey)

        # 평균 밝기보다 밝은 픽셀의 개수 계산
        bright_pixels = np.sum(img_cv_grey > average_brightness)

        # 전체 픽셀 수 계산
        total_pixels = img_cv_grey.size

        # 밝은 픽셀의 비율 계산
        bright_percentage = bright_pixels / total_pixels
        print(f"bright_percentage: {bright_percentage}")

        # 글자가 밝다고 판단되면 이미지 반전
        if bright_percentage > threshold_percentage:  # 평균 밝기보다 밝은 픽셀의 비율이 특정 임계값 이상일 경우
            inverted_image = cv2.bitwise_not(img_cv_grey)  # bitwise_not 함수를 사용하여 반전
            print("이미지 반전 수행")
            return inverted_image
        else:
            print("이미지 반전 수행 안 함")
            return img_cv_grey

    def process_image_for_ocr(self, PIL_img):
        # TODO : Implement using opencv
        temp_filename = self.set_image_dpi(PIL_img)
        im_new = self.remove_noise_and_smooth(temp_filename)
        return im_new

    def set_image_dpi(self, im):
        length_x, width_y = im.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
        # size = (1800, 1800)
        im_resized = im.resize(size, Image.Resampling.LANCZOS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))
        return temp_filename

    def image_smoothening(self, img):
        ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def remove_noise_and_smooth(self, file_name):
        img = cv2.imread(file_name, 0)
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                         3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = self.image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        return or_image

    def remove_noise_and_smooth2(self, img):
        # img = cv2.imread(file_name, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # COLOR_BGR2GRAY, COLOR_RGB2GRAY
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                         3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = self.image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        return or_image
