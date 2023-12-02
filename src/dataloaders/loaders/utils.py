from torch.utils.data import DataLoader

from dataloaders import internet_images
from dataloaders.loaders.internet_images_loader import Loader

from utils.utils import load_module

#yw to note: modify image_size 512 -> 256
def make_dataloader(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, 
                    train=True, batch_size=1, image_size=256, max_iter=1):
    '''
    Args:
        name (`str`): name of the dataset. 
         We recommend to create at least two files for each dataset:
         - `dataloaders/<name>.py` with the `ClothDataset` class definition for this dataset
         - `dataloaders/loaders/<name>_loader.py`  with the `Loader` class definition for this dataset
         - [optional] place your dataset in `samples/<name>/` folder.
    '''
    #import module object
    m_loader = load_module('dataloaders.loaders', name+'_loader') #name=internet_images
    m_dataloader = load_module('dataloaders', name) 
    # m_loader: <module 'dataloaders.loaders.internet_images_loader' from '/home/yw/Desktop/work/point_based_clothing/src/dataloaders/loaders/internet_images_loader.py'>
    # m_dataloader:  <module 'dataloaders.internet_images' from '/home/yw/Desktop/work/point_based_clothing/src/dataloaders/internet_images.py'>


    loader = m_loader.Loader(data_root, rgb_dir, segm_dir, smpl_dir, image_size, smpl_model_path=smpl_model_path)
    # loader : <dataloaders.loaders.internet_images_loader.Loader object at 0x7fc6a571e610>

    dataset = m_dataloader.ClothDataset(loader, datalist)
    # dataset : <dataloaders.internet_images.ClothDataset object at 0x7fc70310fbb0>

    
    if train:
        if isinstance(datalist, list):
            dataset.datalist = dataset.datalist * (max_iter // len(datalist)) * batch_size
        else:  # pd.DataFrame
            dataset.datalist = datalist.loc[datalist.index.repeat((max_iter // len(datalist)) * batch_size)]
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader


def make_dataloaders(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, batch_size, max_iter=1):
    '''
    Args:
        name (`str`): name of the dataset. 
         We recommend to create at least two files for each dataset:
         - `dataloaders/<name>.py` with the `ClothDataset` class definition for this dataset
         - `dataloaders/loaders/<name>_loader.py`  with the `Loader` class definition for this dataset
         - [optional] place your dataset in `samples/<name>/` folder.
    '''
    
    dataloader = make_dataloader(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, 
                                 train=True, batch_size=batch_size, max_iter=max_iter)
    print('dataloader', len(dataloader))
    
    dataloader_val = make_dataloader(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, 
                                     train=False, batch_size=1, max_iter=max_iter)
    print('dataloader_val', len(dataloader_val))
    
    return dataloader, dataloader_val
