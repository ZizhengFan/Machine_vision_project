import pywt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.filters import convolve

# ------------------------------------------------------This is a split line----


def gaussian_filter(image, size=5, sigma=1.5) -> np.ndarray:
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

    blured_image = convolve(image, gaussian_kernel)
    blured_image = cv2.medianBlur(blured_image, ksize=7)

    return blured_image

# 小波变换滤波（根据直方图来搞效果可能会更好）


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

    denoised_image = pywt.idwt2(coeff, 'bior1.3')

    return denoised_image


def log_operation(denoised_image):
    """The logging operation which is used to adjust image's histogram.

    Args:
        denoised_image (np.array): denoised image

    Returns:
        np.array: histogram adjusted image
    """
    img_log = (np.log(denoised_image+1) /
               (np.log(1+np.max(denoised_image)))) * 255
    log_image = np.array(img_log, dtype=np.uint8)

    return log_image

# 使用Sobel算子检测水平和垂直方向梯度


def sobel_gradient(log_image):
    """Two Sobel operators are used to detect the gradient.

    Args:
        log_image (np.array): histogram adjusted image

    Returns:
        np.array: gradient image of x
        np.array: gradient image of y
    """
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobelx = convolve(np.float32(log_image), sx)
    sobely = convolve(np.float32(log_image), sy)

    return sobelx, sobely


def oblique_gradient(log_image):

    ox = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    oy = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])

    obliquex = convolve(np.float32(log_image), ox)
    obliquey = convolve(np.float32(log_image), oy)

    return obliquex, obliquey


def magnitude(sobelx, sobely, threshold_grad=10):
    """From the gradient x and y, calcualte the magnitude and phase.

    Args:
        sobelx (np.array): gradient image of x
        sobely (np.array): gradient image of y
        threshold_grad (int, optional): threshold of gradient. Defaults to 10.
    Returns:
        np.array: the magitude of the gradient
        np.array: the phase of the gradient

    """
    grad = np.sqrt(sobelx**2 + sobely**2)
    phase = cv2.phase(sobelx, sobely, 1)
    # convert to degrees
    phase = (180/np.pi)*phase
    # for those magnitude of gradient lower than 10, we consider it as zero
    x, y = np.where(grad < threshold_grad)
    phase[x, y] = 0
    grad[x, y] = 0

    return grad, phase


def non_max_supression(grad, phase):
    """_summary_

    Args:
        grad (_type_): _description_
        phase (_type_): _description_

    Returns:
        np.array: from new phase, generate new gradient image
    """
    r, c = grad.shape
    # create a new matrix for storing the new phase
    newphase = np.zeros((r, c))
    # 储存过程值
    x1, y1 = np.where(((phase > 0) & (phase <= 22.5)) |
                      ((phase > 157.5) & (phase <= 202.5)) |
                      ((phase > 337.5) & (phase < 360)))

    x2, y2 = np.where(((phase > 22.5) & (phase <= 67.5)) |
                      ((phase > 202.5) & (phase <= 247.5)))

    x3, y3 = np.where(((phase > 67.5) & (phase <= 112.5)) |
                      ((phase > 247.5) & (phase <= 292.5)))

    x4, y4 = np.where(((phase > 112.5) & (phase <= 157.5)) |
                      ((phase > 292.5) & (phase <= 337.5)))

    # 这个new就是新的phase存储的矩阵
    newphase[x1, y1] = 0
    newphase[x2, y2] = 45
    newphase[x3, y3] = 90
    newphase[x4, y4] = 135

    # 设置一个新矩阵newgrad，待会用来根据grad和new生成新的grad
    newgrad = np.zeros((r, c))

    # 非极大值抑制
    for i in range(2, r-2):
        for j in range(2, c-2):
            if newphase[i, j] == 90:
                if((grad[i+1, j] < grad[i, j]) & (grad[i-1, j] < grad[i, j])):
                    newgrad[i, j] = 1

            elif newphase[i, j] == 45:
                if((grad[i+1, j-1] < grad[i, j]) & (grad[i-1, j+1] < grad[i, j])):
                    newgrad[i, j] = 1

            elif newphase[i, j] == 0:
                if((grad[i, j+1] < grad[i, j]) & (grad[i, j-1] < grad[i, j])):
                    newgrad[i, j] = 1

            elif newphase[i, j] == 135:
                if((grad[i+1, j+1] < grad[i, j]) & (grad[i-1, j-1] < grad[i, j])):
                    newgrad[i, j] = 1
    newgrad = np.multiply(newgrad, grad)

    return newgrad


def synthesize_grad(newgrad, log_image):

    obliquex, obliquey = oblique_gradient(log_image)
    oblique_grad = np.sqrt(obliquex**2 + obliquey**2)

    h, w = newgrad.shape
    final_grad = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if newgrad[i, j] >= oblique_grad[i, j]:
                final_grad[i, j] = newgrad[i, j]
            else:
                final_grad[i, j] = oblique_grad[i, j]

    return final_grad


def double_thresholding(newgrad, t_low=0.075, t_high=0.175):
    """_summary_

    Args:
        newgrad (_type_): _description_
        t_low (float, optional): _description_. Defaults to 0.075.
        t_high (float, optional): _description_. Defaults to 0.175.

    Returns:
        _type_: _description_
    """
    # Automating the thresholding selecting process
    r, c = newgrad.shape
    tl = np.uint8(t_low * np.amax(newgrad))
    th = np.uint8(t_high * np.amax(newgrad))
    print(f"The lower threshold is {tl}")
    print(f"The higher threshold is {th}")

    newf = np.zeros((r, c))

    for i in range(2, r-2):
        for j in range(2, c-2):
            if(newgrad[i, j] < tl):
                newf[i, j] = 0
            elif(newgrad[i, j] > th):
                newf[i, j] = 1
            # 判断周围是否有边缘点，若有，则此点也为边缘点
            # 连或8次，有True则True
            elif(newgrad[i+1, j] > th +
                 newgrad[i-1, j] > th +
                 newgrad[i, j+1] > th +
                 newgrad[i, j-1] > th +
                 newgrad[i-1, j-1] > th +
                 newgrad[i-1, j+1] > th +
                 newgrad[i+1, j+1] > th +
                 newgrad[i+1, j-1] > th):
                newf[i, j] = 1

    return np.clip(newf*255, 0, 255).astype(np.uint8)


def canny(image, t_low=0.15, t_high=0.45):
    assert(len(image.shape) == 2)
    blured = gaussian_filter(image)
    denoised = wavelet_filter(blured)
    img_log = log_operation(denoised)
    sobel_x, sobel_y = sobel_gradient(img_log)
    grad, phase = magnitude(sobel_x, sobel_y)
    nms = non_max_supression(grad, phase)
    final_grad = synthesize_grad(nms, img_log)
    edge = double_thresholding(final_grad, t_low, t_high)

    return edge


def morphology_operation(edge_image, morph_size=5):
    morpho_kernel = np.ones((morph_size, morph_size), np.uint8)
    closing = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, morpho_kernel)

    return closing


def feature_detection(closing_image, nfeatures=500):
    feature_operator = cv2.ORB_create(nfeatures=nfeatures)
    # feature_operator = cv2.SIFT_create(nfeatures=nfeatures)

    keypoints, _ = feature_operator.detectAndCompute(closing_image, None)
    featuredImg = cv2.drawKeypoints(closing_image, keypoints, None)

    print("The number of keypoints is:", len(keypoints))

    kps = np.int32([kp.pt for kp in keypoints])
    # print(kps)

    return featuredImg, kps


# ------------------------------------------------------This is a split line----
if __name__ == "__main__":

    path = "crack-detection-opencv-master/Input-Set/RoadCrack_04.jpg"
    src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("The size of this picture is: ", src.shape)

    edge = canny(src, 0.12, 0.3)

    closing = morphology_operation(edge, morph_size=5)

    feature, kps = feature_detection(closing)

    plt.subplot(2, 2, 1)
    plt.imshow(src, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(edge, cmap='gray')
    plt.title('Edge Image')

    plt.subplot(2, 2, 3)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing Image')

    plt.subplot(2, 2, 4)
    plt.imshow(feature, cmap='gray')
    plt.title('Feature Image')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    plt.figure(2)
    plt.plot(kps[:, 0], kps[:, 1], 'o')
    plt.show()
