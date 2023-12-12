import torch_geometric
from torch_geometric.datasets import MNISTSuperpixels, QM7b, GNNBenchmarkDataset, PPI, ZINC
from torch_geometric.loader import DataLoader


def get_dataset(path, ds_choice, device, b_size):
    transform_datasplit = torch_geometric.transforms.RandomNodeSplit(num_val=0.1, num_test=0.1)
    transform_splittrainval = torch_geometric.transforms.RandomLinkSplit(num_val=0.0, num_test=0.15)
    transform_toDevice = torch_geometric.transforms.ToDevice(device=device)

    train_percent = 0.8
    val_percent = 0.1
    test_percetn = 0.1

    if ds_choice == 'benchmark':
        train_dataset = GNNBenchmarkDataset(path, name='PATTERN', split='train')
        val_dataset = GNNBenchmarkDataset(path, name='PATTERN', split='val')
        test_dataset = GNNBenchmarkDataset(path, name='PATTERN', split='test')
        do_embedding = False

    elif ds_choice == 'mnist':
        train_dataset = MNISTSuperpixels(path, train=True)
        val_start = round(len(train_dataset) * (1 - val_percent))

        val_dataset = train_dataset[val_start:]
        train_dataset = train_dataset[:val_start]
        test_dataset = MNISTSuperpixels(path, train=False)

        train_dataset.data = transform_toDevice(train_dataset.data)
        val_dataset.data = transform_toDevice(val_dataset.data)
        test_dataset.data = transform_toDevice(test_dataset.data)

        n_features = train_dataset.num_features
        n_classes = train_dataset.num_classes
        do_embedding = False

    elif ds_choice == 'qm7b':
        train_dataset = QM7b(path)

        n_train = round(len(train_dataset) * 0.8)
        n_val = round(len(train_dataset) * 0.1)
        train_data = train_dataset[:n_train]
        val_data = train_dataset[n_train:n_train+n_val]
        test_data = train_dataset[n_train+n_val:]

        train_dataset = transform_toDevice(train_data.data)
        train_dataset.y = train_dataset.y[:, 0]
        val_dataset = transform_toDevice(val_data.data)
        val_dataset.y = val_dataset.y[:, 0]
        test_dataset = transform_toDevice(test_data.data)
        test_dataset.y = test_dataset.y[:, 0]

        n_features = train_dataset.num_features
        n_classes = 1
        do_embedding = False

    elif ds_choice == 'zinc':
        train_dataset = ZINC(path, subset=True, split='train')
        val_dataset = ZINC(path, subset=True, split='val')
        test_dataset = ZINC(path, subset=True, split='test')

        train_dataset.data = transform_toDevice(train_dataset.data)
        val_dataset.data = transform_toDevice(val_dataset.data)
        test_dataset.data = transform_toDevice(test_dataset.data)

        n_features = train_dataset.num_features
        n_classes = None
        do_embedding = True

    elif ds_choice == 'ppi':
        train_dataset = PPI(path, split='train', )




    train_dl = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=b_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=b_size, shuffle=True)



    return train_dl, val_dl, test_dl, n_features, n_classes, do_embedding