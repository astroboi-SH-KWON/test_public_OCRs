import pytesseract
import easyocr
import os


class publicOCRs:
    def __init__(self):
        pass

    def get_ocr_result(self, img, which_ocr='tesseract', tessdata_prefix=''):
        if which_ocr == 'tesseract':
            os.environ['TESSDATA_PREFIX'] = tessdata_prefix
            return self.get_tesseract_result(img)
        elif which_ocr == 'easyocr':
            return self.get_easyocr_result(img)
        else:
            raise ValueError("Check OCR model.")

    def get_tesseract_result(self, img):
        return pytesseract.image_to_string(img, lang='kor+eng')

    def get_easyocr_result(self, img):
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
        return reader.readtext(img, detail=0)


