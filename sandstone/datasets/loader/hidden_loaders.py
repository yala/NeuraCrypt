from sandstone.datasets.loader.factory import RegisterInputLoader
from sandstone.datasets.loader.abstract_loader import abstract_loader
import numpy as np 
import torch
from sandstone.utils.generic import get_path_for_x

@RegisterInputLoader('default_hiddens_loader')
class HiddensLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path 

    def load_input(self, path, additional):
        '''
        loads hiddens as np array if found, else returns array of zeros
        '''
        zero_vec = np.zeros( self.args.precomputed_hidden_dim)
        hidden =  self.path_to_hidden_dict[path] if path in self.path_to_hidden_dict else zero_vec
        return torch.tensor(hidden)
    
    @property
    def cached_extension(self):
        return '.p'
    
    @property
    def apply_augmentations(self):
        return False

@RegisterInputLoader('numpy_hiddens_loader')
class NumpyHiddensLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        ## Assume that path is keyed by exam
        assert 'exam' in sample
        path = get_path_for_x(sample['exam'], self.args)
                
        return path 

    def load_input(self, path, additional):
        '''
        loads np array
        '''
        return torch.tensor(np.load(path))
    
    @property
    def cached_extension(self):
        return '.npy'
    
    @property
    def apply_augmentations(self):
        return False
