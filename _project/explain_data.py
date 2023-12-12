import torch
import torch_geometric
from torch_geometric.datasets import MNISTSuperpixels, QM7b, GNNBenchmarkDataset, PPI, ZINC
from torch_geometric.loader import DataLoader


def get_dataset(path, ds_choice, device):
    transform_datasplit = torch_geometric.transforms.RandomNodeSplit(num_val=0.1, num_test=0.1)
    transform_splittrainval = torch_geometric.transforms.RandomLinkSplit(num_val=0.0, num_test=0.15)
    transform_toDevice = torch_geometric.transforms.ToDevice(device=device)

    train_percent = 0.8
    val_percent = 0.1
    test_percetn = 0.1
    is_node_classification = False
    do_embedding = False

    if ds_choice == 'benchmark':
        train_dataset = GNNBenchmarkDataset(path, name='PATTERN', split='train')
        val_dataset = GNNBenchmarkDataset(path, name='PATTERN', split='val')
        test_dataset = GNNBenchmarkDataset(path, name='PATTERN', split='test')

        b_size = 32

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
        b_size = 128

    elif ds_choice == 'qm7b':
        init_dataset = QM7b(path)

        n_train = round(len(init_dataset) * 0.8)
        n_val = round(len(init_dataset) * 0.1)
        train_dataset = init_dataset[:n_train]
        val_dataset = init_dataset[n_train:n_train+n_val]
        test_dataset = init_dataset[n_train+n_val:]

        #train_dataset = transform_toDevice(train_data)
        train_dataset.y = train_dataset.y[:, 0]
        train_dataset.x = torch.ones(train_dataset.y.shape[0])
        #val_dataset = transform_toDevice(val_data.data)
        val_dataset.y = val_dataset.y[:, 0]
        val_dataset.x = torch.ones(val_dataset.y.shape[0])
        #test_dataset = transform_toDevice(test_data.data)
        test_dataset.y = test_dataset.y[:, 0]
        test_dataset.x = torch.ones(test_dataset.y.shape[0])

        n_features = 1
        n_classes = 1
        b_size = 32

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
        b_size = 32


    elif ds_choice == 'ppi':
        train_dataset = PPI(path, split='train')
        val_dataset = PPI(path, split='val')
        test_dataset = PPI(path, split='test')

        n_classes = 2

        train_dataset.data = transform_toDevice(train_dataset.data)
        train_dataset.data.y = torch.nn.functional.one_hot(train_dataset.data.y[:, 0].to(torch.long), n_classes)
        val_dataset.data = transform_toDevice(val_dataset.data)
        val_dataset.data.y = torch.nn.functional.one_hot(val_dataset.data.y[:, 0].to(torch.long), n_classes)
        test_dataset.data = transform_toDevice(test_dataset.data)
        test_dataset.data.y = torch.nn.functional.one_hot(test_dataset.data.y[:, 0].to(torch.long), n_classes)

        n_features = train_dataset.num_features
        b_size = 2
        is_node_classification = True




    train_dl = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=b_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=b_size, shuffle=True)



    return train_dl, val_dl, test_dl, n_features, n_classes, do_embedding, is_node_classification