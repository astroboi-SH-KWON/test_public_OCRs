from PIL import Image
import pytesseract
import easyocr
import cv2
import glob
# import os
#
#
# os.environ['TESSDATA_PREFIX'] = "/Users/{User}/anaconda3/pkgs/tesseract-5.3.2-hbe6b26a_2/share/tessdata"

"""
>> tesseract --version                        
tesseract 5.3.2
 leptonica-1.83.1
  libgif 5.2.1 : libjpeg 8d (libjpeg-turbo 3.0.0) : libpng 1.6.40 : libtiff 4.5.1 : zlib 1.2.11 : libwebp 1.3.1 : libopenjp2 2.5.0
 Found NEON
 Found libarchive 3.7.1 zlib/1.2.11 liblzma/5.4.4 bz2lib/1.0.8 liblz4/1.9.4 libzstd/1.5.5
 Found libcurl/8.7.1 SecureTransport (LibreSSL/3.3.6) zlib/1.2.12 nghttp2/1.62.0


>> tesseract --list-langs  
List of available languages in "/Users/{User}/anaconda3/envs/test_ocr/share/tessdata/" 
"""

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'/Users/{User}/anaconda3/pkgs/tesseract-5.3.2-hbe6b26a_2/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = r'/Users/{User}/anaconda3/pkgs/tesseract-5.4.1-h82791c5_0/bin/tesseract'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'


def get_languages(which_ocr='pytesseract'):
    if which_ocr.lower() == 'pytesseract':
        print(pytesseract.get_languages(config='.'))


def get_ocr_result(img_path, which_ocr='pytesseract'):
    if which_ocr.lower() == 'pytesseract':
        tmp_img = Image.open(img_path)
        print(pytesseract.image_to_string(tmp_img, lang='kor+eng'))
    elif which_ocr.lower() == 'easyocr':
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
        img = cv2.imread(img_path)
        text = reader.readtext(img, detail=0)
        for txt in text:
            print(txt)


if __name__ == "__main__":
    img_path_list = glob.glob("./tests/data/*.jpg")
    for img_path in img_path_list:
        which_ocr = 'pytesseract'
        print(f"[{which_ocr}]")
        get_ocr_result(img_path, which_ocr=which_ocr)

        which_ocr = 'easyocr'
        print(f"[{which_ocr}]")
        get_ocr_result(img_path, which_ocr=which_ocr)
        print(f"\n\n")