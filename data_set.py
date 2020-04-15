import torch
import torchvision.transforms as transforms
import pandas as pd
import data_preprocessing as dp

class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        # read csv to memory
        df = pd.read_csv(filename)
        urls = list(df['src'])
        self.urls = urls

    def __getitem__(self, idx):
        # get one url by idx
        return self.urls[idx]

    def __len__(self):
        # return length of whole data set (number of rows)
        return len(self.urls)

def collate_fn(urls):
    sources, targets = dp.filter_images(urls)
    sources = torch.tensor(sources)
    targets = torch.tensor(targets)
    print("sources", sources.shape)
    print("targets", targets.shape)
    return sources, targets


# Uncomment and run 'python data_set.py' to test collate fn:
ds = EdgeDataset('./db/captioned_urls.csv')
dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=2, collate_fn=collate_fn, pin_memory=True)

for src, target in dataloader:
    print("loop")
