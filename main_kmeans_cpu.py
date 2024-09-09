from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch
import os

n_clusters=100
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=128 * 196, n_init="auto")
epochs = 100

for epoch in range(epochs):
    for place1 in range(1):
        for place2 in range(989):
            print(epoch, place1, place2)
            train_features = torch.load(os.path.join("save_features_deit/", f"trainfeat_{place1}_{place2}.pth"))
            kmeans = kmeans.partial_fit(train_features.numpy())
    if epoch % 10 == 0 or epoch == epochs - 1:
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
        torch.save(cluster_centers, os.path.join("KNNcenterDeit", f"epochs{epoch}clusters{n_clusters}.pth"))

print(kmeans.cluster_centers_)


