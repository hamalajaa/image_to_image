import torch
import torchvision.transforms as transforms
import pandas as pd
import data_preprocessing as dp
import skimage.io as io
from torchvision import transforms


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


def collate_fn(imgs):
    sources, targets = dp.filter_images(imgs)
    
    sources = torch.stack(sources)
    targets = torch.stack(targets)
    
    targets = targets.permute(0, 3, 1,2)
    sources = sources.unsqueeze(3).repeat(1, 1, 1, 3)
    sources = sources.permute(0, 3, 1,2).type(torch.FloatTensor)

    print(sources)
    print(sources.shape)
    print(targets.shape)
    
    return sources, targets


# Uncomment and run 'python data_set.py' to test collate fn:
#width = 512
#height = 512
#transform = transforms.Compose([
#    transforms.ToPILImage(),
#    transforms.RandomCrop(
#        (height, width), pad_if_needed=True, padding_mode='constant'),
#    transforms.ToTensor()
#])
#ds = EdgeDataset('./db/captioned_urls.csv', transform=transform)
#dataloader = torch.utils.data.DataLoader(
#    dataset=ds, batch_size=2, collate_fn=collate_fn, pin_memory=True)
#
#for src, target in dataloader:
#
#    print("loop")
    
