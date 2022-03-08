import pywt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve
import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------This is a split line----


def image_read(path) -> np.ndarray:
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img


def log_transform(gray_img) -> np.ndarray:
    """The logging operation which is used to adjust image's histogram.

    Args:
        denoised_image (np.array): denoised image

    Returns:
        np.array: histogram adjusted image
    """
    img_log = (np.log(gray_img+1) /
               (np.log(1+np.max(gray_img)))) * 255
    log_img = np.array(img_log, dtype=np.uint8)

    log_img = gray_img

    return log_img


def filter_noise(log_img, size=5, sigma=1.0) -> np.ndarray:
    """按参数生成高斯卷积核 

    Args:
        size (int, optional): [卷积核的大小]. Defaults to 5.
        sigma (float, optional): [标准差的大小 默认为1即标准正态分布]. Defaults to 1.0.

    Returns:
        [np.array]: [高斯卷积核]
    """
    # 生成二维表格 x为横向的高斯核 y为纵向的高斯核
    x, y = np.mgrid[-(size//2):(size//2)+1, -(size//2):(size//2)+1]
    # 转化为标准差为1的正态分布
    normal = 1 / (2 * np.pi * sigma**2)
    gaussian_kernel = normal * np.exp(-((x**2 + y**2)/(2 * sigma**2)))

    blured_image = convolve(log_img, gaussian_kernel)
    blured_image = cv2.medianBlur(blured_image, ksize=size)

    return blured_image


def wavelet_filter(blured_image, th_wavelet=100, wavelet='haar') -> np.ndarray:
    """wavelet filter

    Args:
        blured_image (np.array): gaussian or median filted image
        th_wavelet (int, optional): the thresold to filter noise. Defaults to 100.
        wavelet (str, optional): define which type of wavelet you use. Defaults to 'haar'.

    Returns:
        np.array: return a furtured filtered image
    """
    coeffs2 = pywt.dwt2(blured_image, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # threshold = 100
    LH_new = pywt.threshold(data=LH, value=th_wavelet,
                            mode='hard', substitute=0)
    HL_new = pywt.threshold(data=HL, value=th_wavelet,
                            mode='hard', substitute=0)
    HH_new = pywt.threshold(data=HH, value=th_wavelet,
                            mode='hard', substitute=0)
    coeff = (LL, (LH_new, HL_new, HH_new))

    denoised_img = pywt.idwt2(coeff, 'bior1.3')

    return denoised_img


def normal_Otsu(denoised_img) -> np.ndarray:
    histogram = cv2.calcHist([denoised_img.astype(np.uint8)], [
                             0], None, [256], [0, 256])
    h, w = denoised_img.shape
    hist_normalize = histogram.ravel()/(h * w)
    Q = hist_normalize.cumsum()
    x_axis = np.arange(256)
    mini = np.inf
    thresh = -1

    for i in range(1, 256):
        # probabilities
        p1, p2 = np.hsplit(hist_normalize, [i])

        # cumulative sum of classes
        q1, q2 = Q[i], Q[255]-Q[i]

        # weights
        b1, b2 = np.hsplit(x_axis, [i])

        # finding means and variances
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < mini:
            mini = fn
            thresh_Otsu = i
    print("the result of normal Otsu's threshold is:", thresh_Otsu)

    return thresh_Otsu


def first_segmentation(denoised_img) -> np.ndarray:
    thresh_Otsu = normal_Otsu(denoised_img)
    mean = np.average(np.uint8(denoised_img.ravel()))
    
    h, w = denoised_img.shape
    for i in range(h):
        for j in range(w):
            if denoised_img[i][j] > thresh_Otsu:
                denoised_img[i][j] = 0

    firstseg_img = denoised_img
    hist_firstseg = cv2.calcHist([firstseg_img.astype(np.uint8)],
                                 [0], None, [256], [0, 256])

    return firstseg_img, hist_firstseg


def iterative_segmentation(firstseg_img, weight=0.25) -> np.ndarray:
    h, w = firstseg_img.shape
    firstseg_img = firstseg_img.astype(np.uint8)
    tmax = firstseg_img.max()
    tmin = firstseg_img.min()
    T0 = np.uint8(np.median(firstseg_img))
    Tk = T0
    print("the initial iterative threshold is: ", T0)

    hist = cv2.calcHist([firstseg_img], [0], None, [256], [0, 256])
    hist_normalize = hist.ravel()/(h * w)

    Q = hist_normalize.cumsum()
    x_axis = np.arange(256)

    for i in range(T0, 255):
        p1, p2 = np.hsplit(hist_normalize, [i])
        q1, q2 = Q[i], Q[255] - Q[i]
        b1, b2 = np.hsplit(x_axis, [i])

        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        temp = weight * (m1 + m2)

        if temp == Tk:
            break
        elif temp == np.inf or temp == -np.inf or temp == np.nan:
            # print("STOP HERE !")
            break
        else:
            Tk = np.uint8(temp)

    # print("Tk is: ", Tk)

    for i in range(h):
        for j in range(w):
            if firstseg_img[i][j] >= Tk or firstseg_img[i][j] == 0:
                firstseg_img[i][j] = 0
            elif firstseg_img[i][j] < Tk:
                firstseg_img[i][j] = 255

    print("the finally iterative threshold is: ", Tk)
    secondseg_img = firstseg_img

    return secondseg_img, Tk


def morphology_operation(secondseg_img, morph_size=5):
    morpho_kernel = np.ones((morph_size, morph_size), np.uint8)
    closing_img = cv2.morphologyEx(
        secondseg_img, cv2.MORPH_CLOSE, morpho_kernel)

    return closing_img


def feature_detection(closing_img, nfeatures=500):
    feature_operator = cv2.ORB_create(nfeatures)
    # feature_operator = cv2.SIFT_create(nfeatures=1500)

    keypoints, _ = feature_operator.detectAndCompute(closing_img, None)
    featured_img = cv2.drawKeypoints(closing_img, keypoints, None)
    print("the number of keypoints is:", len(keypoints))

    kps = np.int32([kp.pt for kp in keypoints])

    return featured_img, kps


# ------------------------------------------------------This is a split line----
if __name__ == '__main__':
    path = "crack-detection-opencv-master/Input-Set/RoadCrack_01.jpg"
    gray_img = image_read(path)
    log_img = log_transform(gray_img)
    blured_image = filter_noise(log_img)
    denoised_img = wavelet_filter(blured_image)
    # first_thresh_Otsu = normal_Otsu(denoised_img)
    firstseg_img, hist_firstseg = first_segmentation(denoised_img)
    secondseg_img, second_thresh_Otsu = iterative_segmentation(firstseg_img)
    closing_img = morphology_operation(secondseg_img)
    featured_img, kps = feature_detection(closing_img)
    
    ret, original_Otsu = cv2.threshold(denoised_img.astype(np.uint8),
                                    0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(f"the opencv Otsu's threshold is {ret}")

# ------------------------------------------------------This is a split line----
    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title("grayscale image")
    plt.subplot(2, 2, 2)
    plt.imshow(log_img, cmap='gray')
    plt.title("log transformation image")
    plt.subplot(2, 2, 3)
    plt.imshow(blured_image, cmap='gray')
    plt.title("blured image")
    plt.subplot(2, 2, 4)
    plt.imshow(denoised_img, cmap='gray')
    plt.title("wavelet denoised image")
    plt.show()

    plt.figure(1, figsize=(7, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(firstseg_img, cmap='gray')
    plt.title("image after first segmentation")
    plt.subplot(1, 2, 2)
    plt.imshow(secondseg_img, cmap='gray')
    plt.title("image after second segmentation")
    plt.show()

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(secondseg_img, cmap='gray')
    plt.title(f"improved Otsu: th={second_thresh_Otsu}")
    plt.subplot(1, 2, 2)
    plt.imshow(~original_Otsu, cmap='gray')
    plt.title(f"original Otsu: th={ret}")
    plt.show()

    plt.figure(3, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(featured_img, cmap='gray')
    plt.title("image and feature points")
    plt.subplot(1, 2, 2)
    plt.plot(kps[:, 0], kps[:, 1], 'o')
    plt.title("the key points")
    plt.show()
