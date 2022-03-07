import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

if __name__ == "__main__":
    # Read input image
    #img = cv2.imread("Input-Set/RoadCrack_02.jpg")
    img = cv2.imread(
        "crack-detection-opencv-master/Input-Set/RoadCrack_03.jpg")
    #img = cv2.imread("Input-Set/my_road_defection_1.png")

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Logarithmic transform
    #img_log = (np.log(gray+1)/(np.log(1+np.max(gray))))*255
    #img_log = np.array(img_log,dtype=np.uint8)
    img_log = gray

    # Calculate histogram
    plt.figure(1)
    plt.subplot(121)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title('Histogram of original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.hist(img_log.ravel(), 256, [0, 256])
    plt.title('Histogram of Log image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Step 1: Improved Otsu thresholding
    ret, otsu_thres = cv2.threshold(
        np.array(img_log, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print("Otsu threshold level:", ret)
    #_, improved_otsu = cv2.threshold(img_log, ret, 255, cv2.THRESH_TRUNC)
    _, improved_otsu = cv2.threshold(img_log, ret, 255, cv2.THRESH_TOZERO_INV)

    plt.figure(2)
    plt.subplot(221)
    plt.imshow(gray, cmap='gray')
    plt.title('Gray image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222)
    plt.imshow(img_log, cmap='gray')
    plt.title('Log image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(otsu_thres, cmap='gray')
    plt.title('Otsu threshold image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    plt.imshow(improved_otsu, cmap='gray')
    plt.title('Improved Otsu threshold image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Step 2: Iterative thresholding
    hist = cv2.calcHist([improved_otsu], [0], None, [256], [0, 256])
    hist[0] = 0
    hist_norm = hist.ravel() / hist.max()
    cumulative = hist.cumsum()
    gray_values = range(256)
    index = np.nonzero(hist)
    #print("nonzero index:",index)
    # Initial values
    plt.figure(3)
    plt.hist(improved_otsu.ravel(), 256, [0, 256])
    plt.show()
    t0 = round(np.median(improved_otsu.ravel()))
    tk = np.zeros(256)
    tk[0] = t0
    #tk[0] = 125
    print(tk)
    print("Initial threshold value:", t0)
    x = np.arange(9.0)
    print(x[0:8])
    print(hist[0:int(tk[0]+1)])
    for i in range(255):
        #p1, p2 = np.hsplit(hist, tk[i]+1)
        #v1, v2 = np.hsplit(gray_values, tk[i]+1)
        p1 = hist[0:int(tk[i]+1)]
        p2 = hist[int(tk[i]+1):256]
        v1 = gray_values[0:int(tk[i]+1)]
        v2 = gray_values[int(tk[i]+1):256]
        print("p1", p1)
        print("v1", v1)
        if cumulative[int(tk[i])] == 0:
            mu1 = 0
        else:
            mu1 = np.sum(p1*v1) / (cumulative[int(tk[i])])
        mu2 = np.sum(p2*v2) / (cumulative[255] - cumulative[int(tk[i])])
        print("mu1", mu1)
        print("mu2", mu2)
        tk[i+1] = round(1 / (2 * (mu1 + mu2)))
        if tk[i+1] == tk[i]:
            thres = tk[i+1]
            break

    print("Iterative thresholds:", tk)

    _, iterative_thres = cv2.threshold(
        improved_otsu, thres, 255, cv2.THRESH_BINARY)

    plt.figure(4)
    plt.subplot(121)
    plt.imshow(improved_otsu, cmap='gray')
    plt.title('Improved Otsu image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(iterative_thres, cmap='gray')
    plt.title('Iterative threhold image'), plt.xticks([]), plt.yticks([])
    plt.show()

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(iterative_thres, kernel, iterations=1)

    # Feature detecting
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(erosion, None)
    kps = np.int32([kp.pt for kp in keypoints])
    featuredImg = cv2.drawKeypoints(erosion, keypoints, None)

    # Plot edges
    plt.figure(5)
    plt.subplot(221), plt.imshow(gray, cmap='gray')
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(improved_otsu, cmap='gray')
    plt.title('Improved Otsu image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(iterative_thres, cmap='gray')
    plt.title('Iterative threshold image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(featuredImg, cmap='gray')
    plt.title('Output image'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.figure(6)
    plt.plot(kps[:, 0], kps[:, 1], 'o')
    plt.show()
