from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch
import os
from torchvision import transforms as pth_transforms
from torchvision import datasets
import argparse

parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
parser.add_argument('--num_workers', default=20, type=int, help='Number of data loading workers per GPU.')
parser.add_argument('--data_path', default='dataset/imagenet100', type=str)
parser.add_argument('--output_dir', default='patchKNNcenter50/', type=str)
parser.add_argument('--n_clusters', default=50, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=100, type=int)
args = parser.parse_args()

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

n_clusters= args.n_clusters
batch_size = args.batch_size
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size * 196, n_init="auto")
epochs = args.epochs
output_dir= args.output_dir

transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    num_workers=args.num_workers,
    pin_memory=False,
    drop_last=True,
)

try:
    os.mkdir(output_dir)
except:
    pass

for epoch in range(epochs):
    for index, samples in enumerate(data_loader_train):
        print(f"{epoch}, {index}/{len(data_loader_train)}", flush=True)
        images = samples[0]
        p = 16
        h = w = images.shape[2] // p
        x = images.reshape(shape=(images.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(images.shape[0] * h * w, p**2 * 3))
        # mean = x.mean(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True)
        # x = (x - mean) / (var + 1.e-6)**.5
        kmeans = kmeans.partial_fit(x.numpy())

    if epoch % 20 == 0 or epoch == epochs - 1:
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
        torch.save(cluster_centers, os.path.join(output_dir, f"epochs{epoch}clusters{n_clusters}.pth"))






