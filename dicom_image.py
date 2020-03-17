import dicom
import numpy as np ; import pandas as pd
import os ; import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.transform import resize
from PIL import Image
import argparse
import sys
import imageio

from glob import glob
parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', dest='mode', default='LDCT', help='LDCT or NDCT')
parser.add_argument('--name', dest = 'name', default='mayo', help = 'mayo or seoul or nbia')
parser.add_argument('--seoul_path', dest='seoul_path', default='/data1/SNUH-denoising/Force LDCT/Force LDCT(241-360)', help='dicom file directory')
parser.add_argument('--mayo_path', dest='mayo_path', default='/data1/mayo-ct', help='dicom file directory')
parser.add_argument('--nbia_path', dest='nbia_path', default='/data1/NBIA/IMG_PAIR', help='dicom file directory')
parser.add_argument('--seoul_input_path', dest='seoul_input_path', default='DE_Thorax 1.0 Br59 M_0.8', help='LDCT image folder name')
parser.add_argument('--seoul_target_path', dest='seoul_target_path', default='DE_Thorax 1.0 Br40 M_0.8', help='NDCT image folder name')
parser.add_argument('--mayo_input_path', dest='mayo_input_path', default='input', help='LDCT image folder name')
parser.add_argument('--mayo_target_path', dest='mayo_target_path', default='target', help='NDCT image folder name')
parser.add_argument('--nbia_input_path', dest='nbia_input_path', default='LDCT', help='LDCT image folder name')
parser.add_argument('--nbia_target_path', dest='nbia_target_path', default='NDCT', help='NDCT image folder name')
parser.add_argument('--extension', dest='extension', default= 'npy', help='extension, [NPY, DCM]')
parser.add_argument('--max_bound', dest='max_bound', type = int, default= 400, help= 'bound')
parser.add_argument('--min_bound', dest='min_bound', type = int, default= -1000, help='bound')
args = parser.parse_args()

if args.name == 'seoul':
    if args.mode == 'LDCT':
        path = np.array(sorted(glob(os.path.join(args.seoul_path,'*',args.seoul_input_path,'*.' + args.extension), recursive=True)))

    else:
        path = np.array(sorted(glob(os.path.join(args.seoul_path, '*', args.seoul_target_path, '*.' + args.extension), recursive=True)))
elif args.name == 'mayo':
    if args.mode == 'LDCT':
        path = np.array(
            sorted(glob(os.path.join(args.mayo_path,args.mayo_input_path, '*.' + args.extension), recursive=True)))

    else:
        path = np.array(
            sorted(glob(os.path.join(args.mayo_path,args.mayo_target_path, '*.' + args.extension), recursive=True)))
else:
    if args.mode == 'nbia':
        path = np.array(
            sorted(glob(os.path.join(args.nbia_path, args.nbia_input_path, '*.' + args.extension), recursive=True)))

    else:
        path = np.array(
            sorted(glob(os.path.join(args.nbia_path, args.nbia_target_path, '*.' + args.extension), recursive=True)))


def Toimage(path,new_spacing=[1,1]):
    startTime = time.time()
    i = 0
    for s in path:
        s = dicom.read_file(s)

        image = s.pixel_array
        image = image.astype(np.int16)

       # slope and intercept
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope

        if slope != 1:
            image = slope * image[slice_number].astype(np.float64)
            image = image[slice_number].astype(np.int16)

        image += np.int16(intercept)


        #spacing
        spacing = np.array(s.PixelSpacing, dtype=np.float32)
        resize_factor = spacing / new_spacing

        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0

        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)*255
        image[image>255] = 255.
        image[image<0] = 0.
        image = resize(image, (256, 256))
        image = np.uint8(image)

        print(image.shape)


        imageio.imwrite('/home/boh001/image/S_{}/{}.png'.format(args.mode,i),image)
        i += 1
        endTime = time.time() - startTime
        print('{} seconds per Process :'.format(endTime))
    print('Done')

def read_npy(path):
    i = 0
    for s in path:

        s = np.load(s).astype(np.int16)
        #MIN_BOUND = s.min()
        #MAX_BOUND = s.max()
        #image = (s - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) * 255
        image = s*(400 -(-1000)) -1000
        print(image[1])
        MIN_BOUND = s.min()
        MAX_BOUND = s.max()
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) * 255

        image[image > 255] = 255.
        image[image < 0] = 0.
        image = resize(image, (256, 256))
        print(image[1])
        image = np.uint8(image)
        if args.name == 'mayo':
            imageio.imwrite('/home/boh001/image/M_{}/{}.png'.format(args.mode, i), image)
        elif args.name == 'nbia':
            imageio.imwrite('/home/boh001/image/N_{}/{}.png'.format(args.mode, i), image)
        i += 1
    print('Done')
if args.extension == 'DCM':
    Toimage(path)
else:
    read_npy(path)