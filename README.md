# test_public_OCRs
Tesseract OCR, easyOCR etc

# References
* https://github.com/madmaze/pytesseract
* https://www.kaggle.com/code/dhorvay/pytesseract-multiple-languages
* https://pyimagesearch.com/2020/08/03/tesseract-ocr-for-non-english-languages/


## 1. Setting up the Environment for silicon mac 
    conda create -n test_ocr python==3.9.18
    conda activate test_ocr

    conda install -c pytorch pytorch==2.5.0  # 2.0.1  # pytorch, pytorchvision 부터 깔기
    conda install -c pytorch torchvision==0.20.0
    pip install easyocr==1.7.2

    conda install conda-forge::tesseract==5.4.1
    conda install -c conda-forge pytesseract==0.3.13

    conda install -c anaconda flask==3.0.3  # ~_api.py 사용 위해. v2.2.2 in mac studio, v2.0.3 in astroboi_m2

    For language pack : https://pyimagesearch.com/2020/08/03/tesseract-ocr-for-non-english-languages/
        os.environ['TESSDATA_PREFIX'] = "/Users/{User}/anaconda3/pkgs/tesseract-5.3.2-hbe6b26a_2/share/tessdata"


## 2. Setting up the Environment for ubuntu 
    conda create -n test_ocr python==3.8.13
    conda activate test_ocr

