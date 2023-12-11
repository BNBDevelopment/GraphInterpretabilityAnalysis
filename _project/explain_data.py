import torch_geometric
from torch_geometric.datasets import MNISTSuperpixels, QM7b, GNNBenchmarkDataset, PPI
from torch_geometric.loader import DataLoader


def get_dataset(path, ds_choice, device, b_size):
    transform_datasplit = torch_geometric.transforms.RandomNodeSplit(num_val=0.1, num_test=0.1)
    transform_splittrainval = torch_geometric.transforms.RandomLinkSplit(num_val=0.0, num_test=0.15)
    transform_toDevice = torch_geometric.transforms.ToDevice(device=device)

    train_percent = 0.8
    val_percent = 0.1
    test_percetn = 0.1

    if ds_choice == 'benchmark':
        train_dataset = GNNBenchmarkDataset(path, name='MNIST', split='train')
        val_dataset = GNNBenchmarkDataset(path, name='MNIST', split='val')
        test_dataset = GNNBenchmarkDataset(path, name='MNIST', split='test')

    elif ds_choice == 'mnist':
        train_dataset = MNISTSuperpixels(path, train=True)
        val_start = round(len(train_dataset) * (1 - val_percent))

        val_dataset = train_dataset[val_start:]
        train_dataset = train_dataset[:val_start]
        test_dataset = MNISTSuperpixels(path, train=False)

    elif ds_choice == 'qm7b':
        train_dataset = QM7b(path)
    elif ds_choice == 'ppi':
        train_dataset = PPI(path, split='train', )

    train_dataset.data = transform_toDevice(train_dataset.data)
    val_dataset.data = transform_toDevice(val_dataset.data)
    test_dataset.data = transform_toDevice(test_dataset.data)

    train_dl = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=b_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=b_size, shuffle=True)

    data = train_dataset[0].to(device)
    n_features = train_dataset.num_features
    n_classes = train_dataset.num_classes

    return train_dl, val_dl, test_dl, n_features, n_classes