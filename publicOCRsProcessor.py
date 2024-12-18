import pytesseract
import easyocr
from easyocr.utils import reformat_input
import os
import cv2
from imutils.perspective import four_point_transform
import imutils
import numpy as np


class publicOCRs:
    """
    https://www.google.com/search?q=%ED%8C%8C%EC%9D%B4%EC%8D%AC+tesseract+%ED%95%99%EC%8A%B5&rlz=1C5CHFA_enKR1061KR1061&oq=&gs_lcrp=EgZjaHJvbWUqCQgBECMYJxjqAjIJCAAQIxgnGOoCMgkIARAjGCcY6gIyCQgCECMYJxjqAjIJCAMQIxgnGOoCMgkIBBAjGCcY6gIyCQgFECMYJxjqAjIJCAYQIxgnGOoCMgYIBxBFGEDSAQoyMzMxNjZqMGo3qAIHsAIB&sourceid=chrome&ie=UTF-8
    easyocr : https://velog.io/@hyunk-go/%ED%81%AC%EB%A1%A4%EB%A7%81-Tesseract-OCR-EasyOCR-OpenCV-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%ED%95%99%EC%8A%B5
    이미지 전처리 : https://yunwoong.tistory.com/76
    https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv
    https://github.com/yardstick17/image_text_reader

    """
    def __init__(self, gpu_flag, pixel_threshold=50):
        self.gpu_flag = False
        if gpu_flag.lower() == 'true':
            self.gpu_flag = True
        self.img_pre = ImagePreprocessor()
        self.pixel_threshold = pixel_threshold

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
            resized = imutils.resize(img, height=max(self.pixel_threshold, img.shape[0]))
            *_, img_cv_grey = self.img_pre.preprocess_image(resized)
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

    def high_pass_filter(self, img_cv_grey, cutoff_frequency=30):
        """이미지에 고주파 필터링을 적용합니다.

        Args:
            image: 이미지
            cutoff_frequency: 차단 주파수 (중심부 마스크 크기 조절)

        Returns:
            고주파 필터링이 적용된 이미지 (NumPy array)
            이미지 로드 실패시 None 반환
        """
        try:
            # 푸리에 변환 수행
            f = np.fft.fft2(img_cv_grey)
            fshift = np.fft.fftshift(f)

            # 고주파 필터 생성 (중심부의 저주파 성분을 마스크 처리)
            rows, cols = img_cv_grey.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

            # 필터 적용
            fshift = fshift * mask

            # 역 푸리에 변환 수행
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)

            # 결과 이미지 정규화 (0-255)
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            return img_back

            # # 고역 통과 필터 커널 생성 (간단한 차연산) eg. Sobel filter, Laplacian filter 등도 high-pass filter의 일종
            # kernel = np.array([[-1, -1, -1],
            #                    [-1, 8, -1],
            #                    [-1, -1, -1]])
            #
            # # 컨볼루션 연산 (필터 적용)
            # filtered_img = cv2.filter2D(img_cv_grey, -1, kernel)
            #
            # return filtered_img

        except:
            raise Exception(f"high_pass_filter:")

    def preprocess_image_binarization(self, img_cv_grey):
        # 대비 향상 (CLAHE 사용)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_cv_grey)

        # 가우시안 블러
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 적응적 이진화
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                       2)  # cv2.THRESH_BINARY

        # 팽창 (선택적) - 텍스트가 끊어져 보이는 경우 사용
        kernel = np.ones((2, 2), np.uint8)  # 커널 크기 조절 가능
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        return dilated

    def is_low_frequency_image(self, img, threshold=0.2):
        """이미지에 low-frequency 성분이 강한지 확인합니다.

        Args:
            img: 이미지
            threshold: low-frequency 성분 판단 기준 (0~1 사이 값)

        Returns:
            bool: low-frequency 성분이 강하면 True, 아니면 False
            None: 이미지 로드 실패 시
        """
        try:

            # 푸리에 변환
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)  # 중심을 중앙으로 이동

            # 중심 부분 영역 추출 (전체 이미지 크기의 20% 정도로 설정)
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.uint8)
            r = 30
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            fshift_masked = np.multiply(fshift, mask)

            # 중심 부분의 평균 밝기 계산
            magnitude_spectrum = 20 * np.log(np.abs(fshift_masked))
            mean_magnitude = np.mean(magnitude_spectrum)

            # 전체 spectrum의 평균 계산
            magnitude_spectrum_all = 20 * np.log(np.abs(fshift))
            mean_magnitude_all = np.mean(magnitude_spectrum_all)

            # 상대적인 비율 계산
            relative_magnitude = mean_magnitude / mean_magnitude_all

            # 임계값과 비교
            if relative_magnitude > threshold:
                return True  # low-frequency 성분이 강함
            else:
                return False  # low-frequency 성분이 약함

        except:
            raise Exception(f"is_low_frequency_image:")

    def preprocess_image_w_contour(self, img, width=200, ksize=(5, 5), min_threshold=75, max_threshold=200):
        img = imutils.resize(img, width=width)
        ratio = img.shape[1] / float(img.shape[1])

        # 이미지를 grayscale로 변환하고 blur를 적용
        # 모서리를 찾기위한 이미지 연산
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, ksize, 0)
        edged = cv2.Canny(blurred, min_threshold, max_threshold)

        image_list_title = ['gray', 'blurred', 'edged']
        image_list = [gray, blurred, edged]

        # contours를 찾아 크기순으로 정렬
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        findCnt = None

        # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
            if len(approx) == 4:
                findCnt = approx
                break

        # 만약 추출한 윤곽이 없을 경우 오류
        if findCnt is None:
            raise Exception("Could not find outline.")

        output = img.copy()
        cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

        image_list_title.append("Outline")
        image_list.append(output)

        # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
        transform_image = four_point_transform(img, findCnt.reshape(4, 2) * ratio)

        # plt_imshow(image_list_title, image_list)
        # plt_imshow("Transform", transform_image)

        return transform_image
