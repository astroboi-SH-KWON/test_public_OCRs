import pytesseract
import easyocr
from easyocr.utils import reformat_input
import os


class publicOCRs:
    def __init__(self, gpu_flag):
        self.gpu_flag = False
        if gpu_flag.lower() == 'true':
            self.gpu_flag = True

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
            return self.get_tesseract_result(img)
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
        Get easyOCR results with cropped image. No need text detection
        :param img: image object(openCV2), image path, image url and etc
        :return: (List) OCR results
        """
        reader = easyocr.Reader(['ko', 'en'], gpu=self.gpu_flag)
        img_pre = ImagePreprocessor()
        *_, img_cv_grey = img_pre.preprocess_image(img)
        return reader.recognize(img_cv_grey)[0][1]


class ImagePreprocessor:
    def __init__(self):
        pass

    def preprocess_image(self, img):
        """
        load and preprocess image to (openCV2) color image for detection, (openCV2) grey image for OCR
        :param img: image object(openCV2), image path, image url and etc
        :return: img_cv_color, img_cv_grey
        """
        img_cv_color, img_cv_grey = reformat_input(img)
        return img_cv_color, img_cv_grey
