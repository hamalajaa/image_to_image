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

    print("list of urls", len(urls))

    sources, targets = dp.filter_images(urls)
    print(sources)
    sources = torch.tensor(sources)
    targets = torch.tensor(targets)
    print("sources", sources.shape)
    print("targets", targets.shape)

    #normalize = transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5, 0.5))
    #sources = normalize(sources)
    #targets = normalize(targets)

    return sources, targets


ds = EdgeDataset('./db/captioned_urls.csv')
dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=2, collate_fn=collate_fn, pin_memory=True)

for src, target in dataloader:
    print("loop")