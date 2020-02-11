# function [output Greg] = dftregistration(buf1,buf2,usfac,sqsz)
# Efficient subpixel image registration by crosscorrelation. This code
# gives the same precision as the FFT upsampled cross correlation in a
# small fraction of the computation time and with reduced memory
# requirements. It obtains an initial estimate of the crosscorrelation peak
# by an FFT and then refines the shift estimation by upsampling the DFT
# only in a small neighborhood of that estimate by means of a
# matrix-multiply DFT. With this procedure all the image points are used to
# compute the upsampled crosscorrelation.
# Manuel Guizar - Dec 13, 2007
#
# For a simpler interface with other programs, inputs and output images are in
# the time domain. To keep the fucntion fast also in the case of big images, it
# only process for registration an inner square in the middle of the input image
# (side size = sqzdefined by squaresize). Translated from Matlab to Python.
# Andrea Console - November 2019
#
# Rewrote all code not authored by either Manuel Guizar or Jim Fienup
# Manuel Guizar - May 13, 2016
#
# Citation for this algorithm:
# Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
# "Efficient subpixel image registration algorithms," Opt. Lett. 33,
# 156-158 (2008).
#
# Inputs
# buf1      Reference image,
#
# buf2      image to register
#
# usfac     Upsampling factor (integer). Images will be registered to
#           within 1/usfac of a pixel. For example usfac = 20 means the
#           images will be registered within 1/20 of a pixel. (default = 1)
#
# Outputs
# output =  [error,diffphase,net_row_shift,net_col_shift]
# error     Translation invariant normalized RMS error between f and g
# diffphase     Global phase difference between the two images (should be
#               zero if images are non-negative).
# net_row_shift net_col_shift   Pixel shifts between images
# Greg      (Optional) registered version of buf2.
#
#
# Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of
# Rochester. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#     * Neither the name of the University of Rochester nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from typing import Union

import numpy as np
import scipy
from scipy import ndimage


def dftregister(buf1: np.ndarray, buf2: np.ndarray, usfac: int):
    def dftups(inmat, nor, noc, usfac=1, roff=0, coff=0):
        # function out=dftups(in,nor,noc,usfac,roff,coff)
        # Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
        # a small region.
        # usfac         Upsampling factor (default usfac = 1)
        # [nor,noc]     Number of pixels in the output upsampled DFT, in
        #               units of upsampled pixels (default = size(in))
        # roff, coff    Row and column offsets, allow to shift the output array to
        #               a region of interest on the DFT (default = 0)
        # Receives DC in upper left corner, image center must be in (1,1)
        # Manuel Guizar - Dec 13, 2007
        # Modified from dftus, by J.R. Fienup 7/31/06

        # This code is intended to provide the same result as if the following
        # operations were performed
        #   - Embed the array "inputmat" in an array that is usfac times larger in each
        #     dimension. ifftshift to bring the center of the image to (1,1).
        #   - Take the FFT of the larger array
        #   - Extract an [nor, noc] region of the result. Starting with the
        #     [roff+1 coff+1] element.

        # It achieves this result by computing the DFT in the output array without
        # the need to zeropad. Much faster and memory efficient than the zero-padded
        # FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]

        nr, nc = np.shape(inmat)
        nor = nor.astype(int)
        noc = noc.astype(int)
        # Compute kernels and obtain DFT by matrix products
        kernc = np.exp(
            (-2j * np.pi / (nc * usfac)) * np.dot(
                np.transpose(np.fft.ifftshift(np.arange(nc))).reshape(nc, 1) - np.floor(nc / 2),
                (np.arange(noc).reshape(1, noc) - coff)))
        kernr = np.exp(
            (-2j * np.pi / (nr * usfac)) * np.dot(np.transpose(np.arange(nor) - roff).reshape(nor, 1),
                                                  (np.fft.ifftshift(np.arange(nr))).reshape(1, nr)
                                                  - np.floor(nr / 2)))
        outmat = np.dot(np.dot(kernr, inmat), kernc)
        return outmat

    # noinspection PyPep8Naming
    def ftpad(imFT, outsize):

        # imFTout = ftpad(imFT,outsize)
        # Pads or crops the Fourier transform to the desired output size. Taking
        # care that the zero frequency is put in the correct place for the output
        # for subsequent FT or IFT. Can be used for Fourier transform based
        # interpolation, i.e. dirichlet kernel interpolation.
        #
        #   Inputs
        # imFT      - Input complex array with DC in [1,1]
        # outsize   - Output size of array [ny nx]
        #
        #   Outputs
        # imFTout   - Output complex image with DC in [1,1]
        # Manuel Guizar - 2014.06.02
        nout = outsize
        nin = np.shape(imFT)
        imFT = np.fft.fftshift(imFT)
        center = np.floor(np.divide(nin, 2)).astype(int)
        imFTout = np.zeros(nout).astype(np.complex)
        centerout = np.floor(np.divide(nout, 2)).astype(int)
        cenout_cen = centerout - center - 1
        imFTout[max(cenout_cen[0], 0): min(cenout_cen[0] + nin[0], nout[1]),
        max(cenout_cen[1], 0): min(cenout_cen[1] + nin[1], nout[1])] \
            = imFT[max(-cenout_cen[0], 0): min(-cenout_cen[0] + nout[0], nin[0]),
              max(-cenout_cen[1], 0): min(-cenout_cen[1] + nout[1], nin[1])]

        imFTout = np.fft.ifftshift(imFTout) * nout[0] * nout[1] / (nin[0] * nin[1])
        return imFTout

    ccmax = row_shift = col_shift = 0
    nr, nc = np.shape(buf2)
    buf1ft = np.fft.fft2(buf1)
    buf2ft = np.fft.fft2(buf2)
    nra = np.fft.ifftshift(np.arange(-np.floor(nr / 2), np.ceil(nr / 2)))
    nca = np.fft.ifftshift(np.arange(-np.floor(nc / 2), np.ceil(nc / 2)))

    if usfac == 0:
        # Simple computation of error and phase difference without registration
        ccmax = np.sum(np.multiply(buf1ft, np.conj(buf2ft)))
        row_shift = col_shift = 0
    elif usfac == 1:
        # Single pixel registration
        cc = np.fft.ifft2(np.multiply(buf1ft, np.conj(buf2ft)))
        ccabs = abs(cc)
        row_shift = np.argmax(np.max(ccabs, axis=1))
        col_shift = np.argmax(np.max(ccabs, axis=0))
        ccmax = cc[row_shift, col_shift] * nr * nc
        # Now change shifts so that they represent relative shifts and not indices
        row_shift = nra[row_shift]
        col_shift = nca[col_shift]
    elif usfac > 1:
        # Start with usfac == 2
        cc = np.fft.ifft2(ftpad(np.multiply(buf1ft, np.conj(buf2ft)), [2 * nr, 2 * nc]))
        ccabs = abs(cc)
        row_shift = np.argmax(np.max(ccabs, axis=1))
        col_shift = np.argmax(np.max(ccabs, axis=0))
        ccmax = cc[row_shift, col_shift] * nr * nc
        # Now change shifts so that they represent relative shifts and not indices
        nra2 = np.fft.ifftshift(np.arange(-np.floor(nr), np.ceil(nr)))
        nca2 = np.fft.ifftshift(np.arange(-np.floor(nc), np.ceil(nc)))
        # Nc2 = ifftshift(-fix(nc):ceil(nc)-1)
        row_shift = nra2[row_shift] / 2
        col_shift = nca2[col_shift] / 2
        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            ### DFT computation ###
            # Initial shift estimate in upsampled grid
            # row_shift = round(row_shift*usfac)/usfac #nonsense @AC
            # col_shift = round(col_shift*usfac)/usfac #nonsense @AC
            dftshift = np.floor(np.ceil(usfac * 1.5) / 2)  # Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            cc = np.conj(dftups(np.multiply(buf2ft, np.conj(buf1ft)), np.ceil(usfac * 1.5),
                                np.ceil(usfac * 1.5), usfac, dftshift - row_shift * usfac,
                                dftshift - col_shift * usfac))
            # Locate maximum and map back to original pixel grid
            ccabs = abs(cc)
            rloc = np.argmax(np.max(ccabs, axis=1))
            cloc = np.argmax(np.max(ccabs, axis=0))
            ccmax = cc[rloc, cloc]
            rloc = rloc - dftshift - 1
            cloc = cloc - dftshift - 1
            row_shift = row_shift + rloc / usfac
            col_shift = col_shift + cloc / usfac

        # If its only one row or column the shift along that dimension has no
        # effect. Set to zero.
        if nr == 1:
            row_shift = 0
        if nc == 1:
            col_shift = 0

    rg00 = np.sum(np.square(abs(buf1ft)))
    rf00 = np.sum(np.square(abs(buf2ft)))
    error = 1.0 - np.square(abs(ccmax)) / (rg00 * rf00)
    error = np.sqrt(abs(error))
    diffphase = np.angle(ccmax)
    return error, diffphase, row_shift, col_shift


def register(buf1: np.ndarray, buf2: np.ndarray, usfac: int, size = 256):
    # calculates the shift between images based on the central part of the images (square side = size)
    # it uses the dftregister function to complete the task
    bigsize = np.shape(buf1)
    left = int(bigsize[0]/2-size/2)
    top = int(bigsize[1]/2-size/2)
    right = int(bigsize[0]/2+size/2)
    bottom = int(bigsize[1]/2+size/2)
    output = dftregister(buf1[left:right, top:bottom], buf2[left:right, top:bottom], usfac)
    return output


def imtrans(image, row_shift, col_shift):
    # shifts the image (2D array) according to the input values
    image = scipy.ndimage.interpolation.shift(image, (row_shift, col_shift), mode='nearest')
    return image


def expaverage(old, new, rate):
    # computes the exponential moving average according to the rate value
    old = new * rate + old * (1 - rate)
    return old


def calibrationaverage(array1, array2):
    # creates the calibration image by combining input images
    return np.minimum(array1, array2)


def imstretch8(image, power=0.33, minclip=0.1, maxclip=99.9, pedestal=10):
    # stretches the input image (2D array), clips extreme values, adds a pedestal
    # output is a 8bit bw image (2D array - values 0..255)
    image = np.power(image, power)
    minp = np.percentile(image, minclip)
    maxp = np.percentile(image, maxclip)
    image = np.clip(image, minp, maxp)
    image = pedestal + (255 - pedestal) * (image - minp) / (maxp - minp)
    return image.astype(np.uint8)


def colorize(image, a_fit, b_fit):
    # adds a_fir and b_t to image (luminance) to reconstruct an RGB file.
    # output is a 3D array
    return image


def sensor2Lab(image):
    # input: a 2D or 3D (RGB) matrix
    # output: 3 matrices. L ~ luminance; a ~ r-g; b ~ b-g
    # if the input is bw, a and b will be 2D arrays of zeros
    # the output Lab is not the official one: for simplicity, it is linear wrt to RGB
    return image, 0, 0
