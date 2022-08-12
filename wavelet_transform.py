import cv2.cv2 as cv
import numpy as np
from pywt import dwt2, wavedec2
import matplotlib.pyplot as plt

im = cv.imread('f:/image/cat.jpg', 0)
if im.shape[1] > 500:
    (h, w) = im.shape[:2]
    width = 500
    height = int(h * (width / float(w)))
    im = cv.resize(im, (width, height), interpolation=cv.INTER_AREA)

# Single-stage decomposition, returns are low-frequency component, a horizontal frequency, vertical frequency, high frequency diagonal correspond respectively to the figure above LL, of HL, LH, HH
cA, (cH, cV, cD) = dwt2(im, 'haar')
# two wavelet decomposition
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = wavedec2(im, 'haar', level=2)

Splicing the respective sub-picture # (cA low frequency range [0,510], the high frequency [-255,255])
AH = np.concatenate([cA, cH+255], axis=1)  # = Axis. 1 represents a column splice
VD = np.concatenate([cV+255, cD+255], axis=1)
res1 = np.concatenate([AH, VD], axis=0)

AH2 = np.concatenate([cA2, cH2+510], axis=1)
VD2 = np.concatenate([cV2+510, cD2+510], axis=1)
A2 = np.concatenate([AH2, VD2], axis=0)
AH1 = np.concatenate([A2, (cH1+255)*2], axis=1)
VD1 = np.concatenate([(cV1+255)*2, (cD1+255)*2], axis=1)
res2 = np.concatenate([AH1, VD1], axis=0)

Display #
plt.figure('2D_DWT_1level')
plt.imshow(res1, cmap='gray', vmin=0, vmax=510)
plt.title('1level')

plt.figure('2D_DWT_2level')
plt.imshow(res2, cmap='gray', vmin=0, vmax=1020)
plt.title('2level')

plt.show()
