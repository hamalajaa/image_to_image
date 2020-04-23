import torch
import torchvision.transforms as transforms
import pandas as pd
import data_preprocessing as dp
import skimage.io as io
from torchvision import transforms
import matplotlib.pyplot as plt


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
            # print(ic)
            img = torch.tensor(io.imread(load_pattern))
            print('Imread')
            print(img.shape)
            img = self.transform(img)
            print('Succesfull')

        except Exception as e:
            print(e)
            print('Error')
        return img

    def __len__(self):
        # return length of whole data set (number of rows)
        return len(self.urls)


def collate_fn(imgs):
    sources, targets = dp.filter_images(imgs)
    # print(sources[0].shape)
    print(targets[0].shape)
    sources = torch.cat(sources)
    targets = torch.cat(targets)

    print("sources", sources.shape)
    print("targets", targets.shape)
    return sources, targets


# Uncomment and run 'python data_set.py' to test collate fn:
width = 512
height = 512
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(
        (height, width), pad_if_needed=True, padding_mode='constant'),
    transforms.ToTensor()
])
ds = EdgeDataset('./db/captioned_urls.csv', transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=2, collate_fn=collate_fn, pin_memory=True)

for src, target in dataloader:
    break
    print(src)
    print(target)
    print("loop")
