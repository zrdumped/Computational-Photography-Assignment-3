import numpy as np
import sys, getopt
from skimage import io
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt
import numba as nb
from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm, writeEXR_1D
from scipy import ndimage
from scipy import signal
import sys, getopt


standardDeviation = 10
# set the side length of the Gaussian filter to be 6 * standardDeviation
GaussianFilterRadius = 3 * standardDeviation 

x_kernel = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
y_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
Lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


#bilateral
bilatetalFilterMethod = 3 #0 basic, 1 joint, 2 denoising with detail transfer, 3 mask-based merging
finalOutput = "data/lamp/final/"
sigma_s = 3
sigma_r = 0.001

sigma_s_flash = 1
sigma_r_flash = 0.05

epsilon = 0.02

tau_shadow = 0.06

#gradient
N = 5000
convergenceE = 0.001
sigma = 40
tau_s = 0.5
tau_ue = 0.4

def saveImage(addr, image):
    image = np.clip(image, 0, 1)
    io.imsave(addr,image)

def generateGaussianFilter(radius, sigma):
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
    filter = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    filter /= filter.sum()
    return filter

def basicBilateralFiltering(ambientImage):
    filterRadius = int(3 * sigma_s)
    spacialWeighting = generateGaussianFilter(filterRadius, sigma_s)

    result = np.zeros_like(ambientImage)
    weight = np.zeros_like(ambientImage)
    for i in range(-filterRadius, filterRadius + 1):
        for j in range(-filterRadius, filterRadius + 1):
            rolledAmbient = np.roll(ambientImage, [i, j], axis=[0,1])
            w_s = spacialWeighting[i + filterRadius, j + filterRadius]

            differenceImage = rolledAmbient - ambientImage
            w_r = np.exp(-(differenceImage * differenceImage) / (2 * sigma_r * sigma_r))

            result += w_s * w_r * rolledAmbient
            weight += w_s * w_r

    return result / weight

def applyDetail(baseImage, flashImage):
    flashBase = basicBilateralFiltering(flashImage)
    return baseImage * (flashImage + epsilon) / (flashBase + epsilon)

def jointBilateralFiltering(ambientImage, flashImage):
    filterRadius = int(3 * sigma_s)
    spacialWeighting = generateGaussianFilter(filterRadius, sigma_s)

    result = np.zeros_like(ambientImage)
    weight = np.zeros_like(ambientImage)
    for i in range(-filterRadius, filterRadius + 1):
        for j in range(-filterRadius, filterRadius + 1):
            #print(str(i) + " " + str(j))
            rolledAmbient = np.roll(ambientImage, [i, j], axis=[0,1])
            w_s = spacialWeighting[i + filterRadius, j + filterRadius]

            rolledFlash = np.roll(flashImage, [i, j], axis=[0,1])
            differenceImage = rolledFlash - flashImage
            w_r = np.exp(-(differenceImage * differenceImage) / (2 * sigma_r * sigma_r))

            result += w_s * w_r * rolledAmbient
            weight += w_s * w_r

    if(not detailTransfer):
        return result / weight
    else:
        return applyDetail(result / weight, flashImage)

def allBilateralFiltering(ambientImage, flashImage):
    filterRadius = int(3 * sigma_s)
    filterRadius_flash = int(3 * sigma_s_flash)

    spacialWeighting = generateGaussianFilter(filterRadius, sigma_s)
    spacialWeighting_flash = generateGaussianFilter(filterRadius_flash, sigma_s_flash)


    basicResult = np.zeros_like(ambientImage)
    basicWeight = np.zeros_like(ambientImage)
    jointResult = np.zeros_like(ambientImage)
    jointWeight = np.zeros_like(ambientImage)
    flashResult = np.zeros_like(ambientImage)
    flashWeight = np.zeros_like(ambientImage)

    for i in range(-filterRadius, filterRadius + 1):
        for j in range(-filterRadius, filterRadius + 1):
            #print(str(i) + " " + str(j))
            rolledAmbient = np.roll(ambientImage, [i, j], axis=[0,1])
            w_s = spacialWeighting[i + filterRadius, j + filterRadius]

            differenceImage = rolledAmbient - ambientImage
            w_r_ambient = np.exp(-(differenceImage ** 2) / (2 * sigma_r * sigma_r))

            rolledFlash = np.roll(flashImage, [i, j], axis=[0,1])
            differenceImage = rolledFlash - flashImage
            w_r_flash = np.exp(-(differenceImage ** 2) / (2 * sigma_r * sigma_r))

            basicResult += w_s * w_r_ambient * rolledAmbient
            basicWeight += w_s * w_r_ambient
            jointResult += w_s * w_r_flash * rolledAmbient
            jointWeight += w_s * w_r_flash

            if(i >= -filterRadius_flash and i <= filterRadius_flash and j >= -filterRadius_flash and j <= filterRadius_flash):
                w_s_flash = spacialWeighting_flash[i + filterRadius_flash, j + filterRadius_flash]
                w_r_flash = np.exp(-(differenceImage ** 2) / (2 * sigma_r_flash * sigma_r_flash))
                flashResult += w_s_flash * w_r_flash * rolledFlash
                flashWeight += w_s_flash * w_r_flash

    flashBase = flashResult / flashWeight
    ambientBase = basicResult / basicWeight
    ambientJoint = jointResult / jointWeight
    flashDetail = (flashImage + epsilon) / (flashBase + epsilon)
    ambientDetail = ambientJoint * flashDetail

    return ambientBase, ambientJoint, ambientDetail, flashDetail

@nb.vectorize([nb.float32(nb.float32)], target='parallel', nopython=True)
def linearizeImage_vectorized(image):
    if(image < 0.0404482):
        return image / 12.92
    else:
        return pow((image + 0.055) / 1.055, 2.4)

@nb.vectorize([nb.float32(nb.float32, nb.float32)], target='parallel', nopython=True)
def generateShadowMask_vectorized(ambientImage, flashImage):
    if(abs(flashImage - ambientImage) < tau_shadow):
        return 1
    else:
        return 0

@nb.vectorize([nb.float32(nb.float32, nb.float32)], target='parallel', nopython=True)
def generateSpecularMask_vectorized(flashImage, maxLuminance):
    if(flashImage > 0.95 * maxLuminance):
        return 1
    else:
        return 0

@nb.vectorize([nb.int16(nb.int16, nb.int16)], target='parallel', nopython=True)
def unionMask_vectorized(ambientImage, flashImage):
    if(ambientImage or flashImage):
        return 1
    else:
        return 0

def generateMask(ambientImage, flashImage):
    mask = np.zeros_like(ambientImage[:,:,0])
    ambientImage_linear = linearizeImage_vectorized(ambientImage)
    flashImage_linear = linearizeImage_vectorized(flashImage)

    ambientImage_luminance = lRGB2XYZ(ambientImage_linear)[:,:,1]
    flashImage_luminance = lRGB2XYZ(flashImage_linear)[:,:,1]

    shadowMask = generateShadowMask_vectorized(ambientImage_luminance, flashImage_luminance)
    shadowMask = ndimage.binary_erosion(shadowMask, iterations = 4).astype(int)
    shadowMask = ndimage.binary_fill_holes(shadowMask)
    shadowMask = ndimage.binary_dilation(shadowMask, iterations = 10).astype(np.int16)

    maxLuminance = np.amax(flashImage_luminance)
    minLuminance = np.amin(flashImage_luminance)
    specularMask = generateSpecularMask_vectorized(flashImage_luminance - minLuminance, maxLuminance - minLuminance)
    specularMask = ndimage.binary_erosion(specularMask).astype(int)
    specularMask = ndimage.binary_fill_holes(specularMask)
    specularMask = ndimage.binary_dilation(specularMask).astype(np.int16)

    mask = unionMask_vectorized(shadowMask, specularMask).astype(np.float32)

    return mask


def generateImages(ambientImage, flashImage):

    outputName = "sigmaS" + str(sigma_s) + "_sigmaR" + str(sigma_r) + "_tau" + str(tau_shadow) + "_sigmaSFlash" + str(sigma_s_flash) + "_sigmaRFlash" + str(sigma_r_flash) + ".jpg"

    mask = generateMask(ambientImage, flashImage)
    saveImage(finalOutput + "bilateralMask" + outputName, np.dstack((mask, mask, mask)))
    #plt.imshow(np.dstack((mask, mask, mask)))
    #plt.show()

    #basic
    rBase, rJoint, rDetail, rFlashDetail = allBilateralFiltering(ambientImage[:,:,0],flashImage[:,:,0])
    gBase, gJoint, gDetail, gFlashDetail = allBilateralFiltering(ambientImage[:,:,1],flashImage[:,:,1])
    bBase, bJoint, bDetail, bFlashDetail = allBilateralFiltering(ambientImage[:,:,2],flashImage[:,:,2])
    saveImage(finalOutput + "basicBilateral" + outputName, np.dstack((rBase, gBase, bBase)))
    saveImage(finalOutput + "jointBilateral" + outputName, np.dstack((rJoint, gJoint, bJoint)))
    saveImage(finalOutput + "detailBilateral" + outputName, np.dstack((rDetail, gDetail, bDetail)))
    saveImage(finalOutput + "flashDetailBilateral" + outputName, np.dstack((rFlashDetail, gFlashDetail, bFlashDetail)))
    rMask = (1 - mask) * rDetail + mask * rBase
    gMask = (1 - mask) * gDetail + mask * gBase
    bMask = (1 - mask) * bDetail + mask * bBase
    saveImage(finalOutput + "maskBilateral" + outputName, np.dstack((rMask, gMask, bMask)))

    return

def bilateralFilter():
    #ambientImage = io.imread("data/lamp/lamp_ambient.tif").astype(np.float32) / 255
    #flashImage = io.imread("data/lamp/lamp_flash.tif").astype(np.float32) / 255
    #mask = generateMask(ambientImage, flashImage)
    #image = np.dstack((mask,mask,mask))
    #plt.imshow(image)
    #plt.show()
    generateImages()
    return

def PoissonSolver(image, gradient_x, gradient_y, initialization = False):
    Ixx = signal.convolve2d(gradient_x, x_kernel, mode="same")
    Iyy = signal.convolve2d(gradient_y, y_kernel, mode="same")
    div = Ixx + Iyy

    print("reintegrate")
    # initialization
    I_star = image
    mask = np.ones_like(image)

    (rows, cols) = np.shape(image)
    mask[0,:] = 0
    mask[:,0] = 0
    mask[rows - 1,:] = 0
    mask[:,cols - 1] = 0
    if (not initialization):
        I_star *= 1 - mask

    
    r = div - signal.convolve2d(I_star, Lap_kernel, mode="same")
    d = r
    delta_new = np.dot(r.flatten(), r.flatten())
    r_norm = delta_new ** 0.5
    n = 0

    while (r_norm > convergenceE and n < N):
        q = signal.convolve2d(d, Lap_kernel, mode="same")
        eta = delta_new / np.dot(d.flatten(), q.flatten())
        I_star = I_star + mask * (eta * d)
        r = r - eta * q
        delta_old = delta_new
        delta_new = np.dot(r.flatten(), r.flatten())
        beta = delta_new / delta_old
        d = r + beta * d
        n = n + 1
        r_norm = delta_new ** 0.5


    print(str(n) + " " + str(r_norm))

    return I_star, n

def JacobiPreconditioning(image, gradient_x, gradient_y):
    Ixx = signal.convolve2d(gradient_x, x_kernel, mode="same")
    Iyy = signal.convolve2d(gradient_y, y_kernel, mode="same")
    div = Ixx + Iyy

    print("reintegrate")
    # initialization
    I_star = np.zeros_like(image)
    mask = np.ones_like(image)

    (rows, cols) = np.shape(image)
    mask[0,:] = 0
    mask[:,0] = 0
    mask[rows - 1,:] = 0
    mask[:,cols - 1] = 0
    I_star = (1 - mask) * image

    r = div - signal.convolve2d(I_star, Lap_kernel, mode="same")
    d = -0.25 * r
    delta_new = np.dot(r.flatten(), d.flatten())
    n = 0
    r_norm = np.dot(r.flatten(), r.flatten()) ** 0.5

    while (r_norm > convergenceE and n < N):
        q = signal.convolve2d(d, Lap_kernel, mode="same")
        eta = delta_new / np.dot(d.flatten(), q.flatten())
        I_star = I_star + mask * (eta * d)
        r = r - eta * q
        s = -0.25 * r
        delta_old = delta_new
        delta_new = np.dot(r.flatten(), s.flatten())
        beta = delta_new / delta_old
        d = s + beta * d
        n = n + 1
        r_norm = np.dot(r.flatten(), r.flatten()) ** 0.5

    print(n)

    return I_star

def differentiateAndReintegrate(image):
    print("differentiate")
    Ix = signal.convolve2d(image, x_kernel, mode="same")
    Iy = signal.convolve2d(image, y_kernel, mode="same")

    return PoissonSolver(image, Ix, Iy)


@nb.vectorize([nb.float64(nb.float64,nb.float64)], target='parallel', nopython=True)
def devide_vectorized(n, d):
    if(d == 0):
        return 0
    else:
        return n / d

def calculateGradientField(ambientImage, flashImage):
    alpha_x = signal.convolve2d(ambientImage, x_kernel, mode="same")
    alpha_y = signal.convolve2d(ambientImage, y_kernel, mode="same")

    phi_x = signal.convolve2d(flashImage, x_kernel, mode="same")
    phi_y = signal.convolve2d(flashImage, y_kernel, mode="same")

    M_numerator = np.abs(phi_x * alpha_x + phi_y * alpha_y)
    M_dominator = np.power(phi_x ** 2 + phi_y ** 2, 0.5) * np.power(alpha_x**2 + alpha_y ** 2, 0.5)
    M = devide_vectorized(M_numerator, M_dominator)


    weighting_s = np.tanh(sigma * (flashImage - tau_s))
    weighting_min = np.amin(weighting_s)
    weighting_max = np.amax(weighting_s)
    weighting_s = (weighting_s - weighting_min) / (weighting_max - weighting_min)

    phi_x_star = weighting_s * alpha_x + (1 - weighting_s) * (M * phi_x + (1 - M) * alpha_x)
    phi_y_star = weighting_s * alpha_y + (1 - weighting_s) * (M * phi_y + (1 - M) * alpha_y)

    return phi_x_star, phi_y_star, alpha_x, alpha_y, phi_x, phi_y

def fuseGradientField(ambientImage, flashImage):

    #ambientImage = tuple(pyramid_gaussian(ambientImage, downscale=2, max_layer = 5))[4]
    #flashImage = tuple(pyramid_gaussian(flashImage, downscale=2, max_layer = 5))[4]

    phi_x_star, phi_y_star, alpha_x, alpha_y, phi_x, phi_y = calculateGradientField(ambientImage, flashImage)

    return PoissonSolver(ambientImage, phi_x_star, phi_y_star),phi_x_star, phi_y_star, alpha_x, alpha_y, phi_x, phi_y

def generateGradientImages(ambientImage, flashImage):

    rphi_x_star, rphi_y_star, ralpha_x, ralpha_y, rphi_x, rphi_y = calculateGradientField(ambientImage[:,:,0],flashImage[:,:,0])
    gphi_x_star, gphi_y_star, galpha_x, galpha_y, gphi_x, gphi_y = calculateGradientField(ambientImage[:,:,1],flashImage[:,:,1])
    bphi_x_star, bphi_y_star, balpha_x, balpha_y, bphi_x, bphi_y = calculateGradientField(ambientImage[:,:,2],flashImage[:,:,2])

    saveImage(finalOutput + "alphaGradientRX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((ralpha_x, ralpha_x, ralpha_x)))
    saveImage(finalOutput + "alphaGradientRY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((ralpha_y, ralpha_y, ralpha_y)))
    saveImage(finalOutput + "phiGradientRX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((rphi_x, rphi_x, rphi_x)))
    saveImage(finalOutput + "phiGradientRY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((rphi_y, rphi_y, rphi_y)))
    saveImage(finalOutput + "phiStarGradientRX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((rphi_x_star, rphi_x_star, rphi_x_star)))
    saveImage(finalOutput + "phiStarGradientRY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((rphi_y_star, rphi_y_star, rphi_y_star)))
    
    saveImage(finalOutput + "alphaGradientGX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((galpha_x, galpha_x, galpha_x)))
    saveImage(finalOutput + "alphaGradientGY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((galpha_y, galpha_y, galpha_y)))
    saveImage(finalOutput + "phiGradientGX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((gphi_x, gphi_x, gphi_x)))
    saveImage(finalOutput + "phiGradientGY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((gphi_y, gphi_y, gphi_y)))
    saveImage(finalOutput + "phiStarGradientGX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((gphi_x_star, gphi_x_star, gphi_x_star)))
    saveImage(finalOutput + "phiStarGradientGY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((gphi_y_star, gphi_y_star, gphi_y_star)))

    saveImage(finalOutput + "alphaGradientBX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((balpha_x, balpha_x, balpha_x)))
    saveImage(finalOutput + "alphaGradientBY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((balpha_y, balpha_y, balpha_y)))
    saveImage(finalOutput + "phiGradientBX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((bphi_x, bphi_x, bphi_x)))
    saveImage(finalOutput + "phiGradientBY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((bphi_y, bphi_y, bphi_y)))
    saveImage(finalOutput + "phiStarGradientBX_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((bphi_x_star, bphi_x_star, bphi_x_star)))
    saveImage(finalOutput + "phiStarGradientBY_sigma" + str(sigma) + "_tau" + str(tau_s) + ".jpg", np.dstack((bphi_y_star, bphi_y_star, bphi_y_star)))

    r, nr = PoissonSolver(ambientImage[:,:,0], rphi_x_star, rphi_y_star)
    g, ng = PoissonSolver(ambientImage[:,:,1], gphi_x_star, gphi_y_star)
    b, nB = PoissonSolver(ambientImage[:,:,2], bphi_x_star, bphi_y_star)
    saveImage(finalOutput + "gradientMerge_ambientInit_sigma" + str(sigma) + "_tau" + str(tau_s) + "_nr" + str(nr) + "_ng" + str(ng) + "_nb" + str(nB) + ".jpg", np.dstack((r, g, b)))

    r, nr = PoissonSolver(flashImage[:,:,0], rphi_x_star, rphi_y_star)
    g, ng = PoissonSolver(flashImage[:,:,1], gphi_x_star, gphi_y_star)
    b, nB = PoissonSolver(flashImage[:,:,2], bphi_x_star, bphi_y_star)
    saveImage(finalOutput + "gradientMerge_flashInit_sigma" + str(sigma) + "_tau" + str(tau_s) + "_nr" + str(nr) + "_ng" + str(ng) + "_nb" + str(nB) + ".jpg", np.dstack((r, g, b)))

    r, nr = PoissonSolver((ambientImage[:,:,0] + flashImage[:,:,0]) / 2.0, rphi_x_star, rphi_y_star)
    g, ng = PoissonSolver((ambientImage[:,:,1] + flashImage[:,:,1]) / 2.0, gphi_x_star, gphi_y_star)
    b, nB = PoissonSolver((ambientImage[:,:,2] + flashImage[:,:,2]) / 2.0, bphi_x_star, bphi_y_star)
    saveImage(finalOutput + "gradientMerge_averageInit_sigma" + str(sigma) + "_tau" + str(tau_s) + "_nr" + str(nr) + "_ng" + str(ng) + "_nb" + str(nB) + ".jpg", np.dstack((r, g, b)))

    r, nr = PoissonSolver(np.zeros_like(ambientImage[:,:,0]), rphi_x_star, rphi_y_star)
    g, ng = PoissonSolver(np.zeros_like(ambientImage[:,:,0]), gphi_x_star, gphi_y_star)
    b, nB = PoissonSolver(np.zeros_like(ambientImage[:,:,0]), bphi_x_star, bphi_y_star)
    saveImage(finalOutput + "gradientMerge_zeroInit_sigma" + str(sigma) + "_tau" + str(tau_s) + "_nr" + str(nr) + "_ng" + str(ng) + "_nb" + str(nB) + ".jpg", np.dstack((r, g, b)))

def MyGenerateGradientImages(ambientImage, flashImage):

    ambientImage = tuple(pyramid_gaussian(ambientImage, downscale=2, max_layer = 3))[2]
    flashImage = tuple(pyramid_gaussian(flashImage, downscale=2, max_layer = 3))[2]

    phi_x_star, phi_y_star, alpha_x, alpha_y, phi_x, phi_y = calculateGradientField(ambientImage,flashImage)

    return PoissonSolver(ambientImage, phi_x_star, phi_y_star)

def multiGridFuse(ambientImage, flashImage):

    ambient_pyramid = tuple(pyramid_gaussian(ambientImage, downscale=2))
    flash_pyramid = tuple(pyramid_gaussian(flashImage, downscale=2))

    imageNum = len(ambient_pyramid)

    I_star = ambient_pyramid[imageNum - 1]

    for i in range(imageNum):
        print(i)
        ambient_downsampled = ambient_pyramid[imageNum - i - 1]
        flash_downsampled = flash_pyramid[imageNum - i - 1]

        phi_x_star, phi_y_star, alpha_x, alpha_y, phi_x, phi_y = calculateGradientField(ambient_downsampled, flash_downsampled)

        if (i == 0):
            I_star, n = PoissonSolver(I_star, phi_x_star, phi_y_star)
        else:
            I_star, n = PoissonSolver(I_star, phi_x_star, phi_y_star, True)

        if (i < imageNum - 1):
            I_star = np.resize(I_star, np.shape(ambient_pyramid[imageNum - i - 2]))

        print("grid" + str(n))

    return I_star

def reflectionRemoval(ambientImage, flashImage):

    ambientImage = tuple(pyramid_gaussian(ambientImage, downscale=2, max_layer = 3))[2]
    flashImage = tuple(pyramid_gaussian(flashImage, downscale=2, max_layer = 3))[2]

    H = ambientImage + flashImage
    H_min = np.amin(H)
    H_max = np.amax(H)
    I = (H - H_min) / (H_max - H_min)
    weighting_ue = 1 - np.tanh(sigma * (I - tau_ue))
    weighting_min = np.amin(weighting_ue)
    weighting_max = np.amax(weighting_ue)
    weighting_ue = (weighting_ue - weighting_min) / (weighting_max - weighting_min)

    H_x = signal.convolve2d(H, x_kernel, mode="same")
    H_y = signal.convolve2d(H, y_kernel, mode="same")

    alpha_x = signal.convolve2d(ambientImage, x_kernel, mode="same")
    alpha_y = signal.convolve2d(ambientImage, y_kernel, mode="same")

    H_x_proj = alpha_x * (np.dot(np.abs(H_x.flatten()), np.abs(alpha_x.flatten())) / np.dot(alpha_x.flatten(), alpha_x.flatten()))
    H_y_proj = alpha_y * (np.dot(np.abs(H_y.flatten()), np.abs(alpha_y.flatten())) / np.dot(alpha_y.flatten(), alpha_y.flatten()))

    phi_x_star = weighting_ue * H_x + (1 - weighting_ue) * H_x_proj
    phi_y_star = weighting_ue * H_y + (1 - weighting_ue) * H_y_proj

    return PoissonSolver(H, phi_x_star, phi_y_star)


def main(argv):
    global finalOutput
    global sigma_s
    global sigma_r
    global sigma_s_flash
    global sigma_r_flash
    global tau_shadow
    # gradient
    global N
    global convergenceE
    global sigma
    global tau_s
    global tau_ue

    mergeMethod = 0 #0 bilateral, 1 gradient, 2 grid, 3 reflection removal

    ambientAddr = "data/lamp/lamp_ambient.tif"
    flashAddr = "data/lamp/lamp_flash.tif"

    try:
        opts, args = getopt.getopt(argv,"i:f:o:m:ss:sr:ssf:srf:ts:N:e:s:tue",["ifile=","flashfile=","ofile=","method=",
                                                              "sigmaS=", "sigmaR=",
                                                              "sigmaFlashS=","sigmaFlashR=","tauS=",
                                                              "N=","convergenceE=","sigma=","tauUe="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            ambientAddr = arg
        elif opt in ("-f", "--flashfile"):
            flashAddr = arg        
        elif opt in ("-o", "--ofile"):
            finalOutput = arg
        elif opt in ("-m", "--method"):
            mergeMethod = arg        
        elif opt in ("-ss", "--sigmaS"):
            sigma_s = arg        
        elif opt in ("-sr", "--sigmaR"):
            sigma_r = arg
        elif opt in ("-ssf", "--sigmaFlashS"):
            sigma_s_flash = arg        
        elif opt in ("-srf", "--sigmaFlashR"):
            sigma_r_flash = arg   
        elif opt in ("-ts", "--tauS"):
            tau_shadow = arg   
            tau_s = arg
        elif opt in ("-N", "--N"):
            N = arg          
        elif opt in ("-e", "--convergenceE"):
            convergenceE = arg          
        elif opt in ("-s", "--sigma"):
            sigma = arg         
        elif opt in ("-tue", "--tauUe"):
            tau_ue = arg   

    ambientImage = io.imread(ambientAddr).astype(np.float32) / 255
    flashImage = io.imread(flashAddr).astype(np.float32) / 255
    if (mergeMethod == 0):
        generateImages(ambientImage, flashImage)
    elif (mergeMethod == 1):
        generateGradientImages(ambientImage, flashImage)
    elif (mergeMethod == 2):
        r = multiGridFuse(ambientImage[:,:,0], flashImage[:,:,0])
        g = multiGridFuse(ambientImage[:,:,1], flashImage[:,:,1])
        b = multiGridFuse(ambientImage[:,:,2], flashImage[:,:,2])
        saveImage(finalOutput + "gridFuse_sigma" + str(sigma) + "_tau" + str(tau_s) +  ".jpg", np.dstack((r, g, b)))
    elif (mergeMethod == 3):
        r, n = reflectionRemoval(ambientImage[:,:,0], flashImage[:,:,0])
        g, n = reflectionRemoval(ambientImage[:,:,1], flashImage[:,:,1])
        b, n = reflectionRemoval(ambientImage[:,:,2], flashImage[:,:,2])
        saveImage(finalOutput + "reflectionRemoval_sigma" + str(sigma) + "_tauUE" + str(tau_ue) + ".jpg", np.dstack((r, g, b)))


if __name__ == "__main__":
    main(sys.argv[1:])
