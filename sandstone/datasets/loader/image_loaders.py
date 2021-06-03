from sandstone.datasets.loader.factory import RegisterInputLoader
from sandstone.datasets.loader.abstract_loader import abstract_loader
import cv2
import numpy as np
import pydicom
import os

LOADING_ERROR = 'LOADING ERROR! {}'
IRS_IMAGE_SIZE = (384, 288)
@RegisterInputLoader('default_image_loader')
class OpenCVLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        '''
        loads as grayscale image
        '''
        return cv2.imread(path, 0)

    @property
    def cached_extension(self):
        return '.png'

@RegisterInputLoader('color_image_loader')
class OpenCVColorLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        '''
        loads as colored image
        '''
        return cv2.imread(path, 1)

    @property
    def cached_extension(self):
        return '.png'

@RegisterInputLoader('numpy_image_loader')
class NumpyImgLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        '''
        additional must include 'path_format' and 'path_format_values' keys, e.g.:
            additional['path_format'] = 'chan={}_{}'
            additional['path_format_values'] = [1, path]
        '''
        if self.args.load_img_as_npz:
            if 'use_formatted_path' in additional and additional['use_formatted_path']:
                return additional['path_format'].format(*additional['path_format_values'])
            else:
                if not path.endswith('.npz') and not self.args.load_IRS_as_npy:
                    return "{}.npz".format(path)
        return path

    def load_input(self, path, additional):
        if self.args.load_img_as_npz:
            arr = np.load(path)
            # IRS as Numpy
            if self.args.load_IRS_as_npy:
                num_frames, WH = arr.shape
                WH_size = list(reversed( IRS_IMAGE_SIZE))
                arr = arr.reshape([num_frames, *WH_size]).transpose([0,2,1])
                frame_idx = []
                for f in additional['frames']:
                    frame_idx.append( int(f*num_frames))
                arr = arr[np.array(frame_idx)]
                arr = arr.transpose([1,2,0])
                return arr

            try:
                return arr['sample']
            except KeyError:
                raise Exception(LOADING_ERROR.format('NPZ file expects \'sample\' as variable name to load numpy array.'))
        else:
            return np.load(path)

    @property
    def cached_extension(self):
        return '.npz'

@RegisterInputLoader('dicom_loader')
class DicomLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        try:
            dcm_object = pydicom.dcmread(path)
        except:
             raise Exception(LOADING_ERROR.format('COULD NOT LOAD DICOM.'))
        arr =  dcm_object.pixel_array.astype(np.int16)
        arr = arr.reshape( [*arr.shape, 1])
        return arr

    @property
    def cached_extension(self):
        return '.dcm'