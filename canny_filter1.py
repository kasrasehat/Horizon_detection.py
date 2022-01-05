from scipy import ndimage
import  numpy as np
import cv2
#from numba import jit


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    img = img.astype(float)
    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio, highThresholdRatio):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = img.max() * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.uint8(25)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def edge_channel(frame):
    #frame = cv2.medianBlur(frame, 11)
    gaussian = gaussian_kernel(size=30, sigma=2.5)
    #frame = cv2.filter2D(frame, -1, gaussian)
    #frame = cv2.bilateralFilter(frame, 5, 170, 150)
    frame = cv2.resize(frame, (192, 108))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gaussian = gaussian_kernel(size= 25 ,sigma= 3)
    frame = cv2.filter2D(frame, -1, gaussian)
    grad, angle1 = sobel_filters(frame)
    angle = ((255 * (angle1 + np.pi)/(2 * np.pi)))
    mask = (grad > 60)
    mask = mask.astype(float)
    angle = np.multiply(mask, angle)
    angle = angle.astype(np.uint8)
    grad = non_max_suppression(grad ,angle1)
    grad, weak, strong = threshold(grad, lowThresholdRatio = 0.04, highThresholdRatio = 0.16)
    grad = grad.astype(np.uint8)
    grad = hysteresis(grad, weak, strong)
    #grad = cv2.resize(grad, (1920, 1080))
    #angle = cv2.resize(angle, (1920, 1080))

    return grad, angle


