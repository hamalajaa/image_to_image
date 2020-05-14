import torch
import torchvision.transforms as transforms
import pandas as pd
import data_preprocessing as dp
import skimage.io as io
import os
from torchvision import transforms

"""
BAM dataset, images from URLs
"""
class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform):
        # read csv to memory
        df = pd.read_csv(filename)
        urls = list(df['src'])
        self.urls = urls
        self.transform = transform

    def __getitem__(self, idx):
        # get one url by idx
        load_pattern = self.urls[idx]
        img = torch.zeros((512, 512, 3))
        try:
            img_cand = io.imread(load_pattern)
            img_cand = self.transform(img_cand)
            
            # C * W * H -> W * H * C
            img_cand = img_cand.permute(1,2,0)

            if img_cand.shape[2] == 3:
                img = img_cand
            
        except Exception as e:
            print(e)
            print('Error')
        return img

    def __len__(self):
        # return length of whole data set (number of rows)
        return len(self.urls)

"""
CUB2011
"""
class BirdEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform):
        # read images to memory as np arrays
        file_spec = '*.jpg'
        file_path = os.path.join(filename, file_spec)
        self.images = io.imread_collection(file_path)
        self.transform = transform


    def __getitem__(self, idx):
        # get one image as np array
        image = self.images[idx]
        image = self.transform(image)
        # C * W * H -> W * H * C
        image = image.permute(1,2,0)
        return image

    def __len__(self):
        # return length of whole data set (number of rows)
        return len(self.images)


def collate_fn(imgs):
    """
    Outputs sources and targets of shape [batch_size, n_channels, width, height]
    """
    sources, targets = dp.filter_images(imgs)
    
    sources = torch.stack(sources)
    targets = torch.stack(targets)
    #print(sources.shape)
    #print(targets.shape)
    targets = targets.repeat(1, 1, 1, 3)
    targets = targets.permute(0, 3, 1, 2)
    
    sources = sources.unsqueeze(3).repeat(1, 1, 1, 3)
    #sources = sources.permute(0, 3, 1, 2).type(torch.FloatTensor)
    sources = sources.permute(0, 3, 1, 2).type(torch.FloatTensor)
    #print(sources)
    #print(sources.shape)
    #print(targets.shape)
    
    return sources, targets

class BirdBWDataset(torch.utils.data.Dataset):
    def __init__(self, filename, transform):
        # read images to memory as np arrays
        file_spec = '*.jpg'
        file_path = os.path.join(filename, file_spec)
        self.images = io.imread_collection(file_path)
        self.transform = transform
        self.bw_transform = transforms.Grayscale(num_output_channels=1)


    def __getitem__(self, idx):
        # get one image as np array
        image = self.images[idx]
        image = self.transform(image)
        
        # Create input image (target as black and white)
        bw_image = self.bw_transform(image)
        
        to_tensor = transforms.ToTensor()
        
        image = to_tensor(image)

        # C * W * H -> W * H * C
        #image = image.permute(1, 2, 0)
        
        bw_image = to_tensor(bw_image)

        # transofrm to C * W * H -> W * H * C
        # and repeat the only channel to match input
        #bw_image = bw_image.permute(1, 2, 0)
        bw_image = bw_image.repeat(3, 1, 1)
        
        # image: target
        # bw_image: input image
        return bw_image, image

    def __len__(self):
        # return length of whole data set (number of rows)
        return len(self.images)



# Uncomment and run 'python data_set.py' to test collate fn:
width = 512
height = 512
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(
        (height, width), pad_if_needed=True, padding_mode='constant')
    #transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor()
])
#ds = BirdEdgeDataset('./db/birds/dummy/', transform=transform)
#dataloader = torch.utils.data.DataLoader(
#    dataset=ds, batch_size=2, collate_fn=collate_fn, pin_memory=True)

ds = BirdBWDataset('./db/birds/dummy/', transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=2, pin_memory=True)

for src, target in dataloader:
    #print("data set testing loop", src.sum(), target.sum())
    print(src.shape, target.shape)
    break    #