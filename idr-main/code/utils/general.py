import os
from glob import glob
import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, device):
    '''
     Split the input to fit memory for large resolution.
     Adjust 'n_pixels' if you encounter out of memory errors.
    '''
    n_pixels = 10000  # Adjust this based on your memory availability
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).to(device), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx.to(device))
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx.to(device))
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size, device):
    ''' Merge the split output. '''
    
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1).to(device) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]).to(device) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs