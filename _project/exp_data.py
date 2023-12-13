import torch
import torch_geometric
from torch_geometric.datasets import MoleculeNet, QM7b, ZINC, PPI, MNISTSuperpixels
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance, Cartesian


def getBinaryClassifier(batch_size):
    ds = MoleculeNet("data/binclass", name='ClinTox')
    #bad_idxs = []
    # for idx, item in enumerate(ds):
    #     if 0 in item.x.shape:
    #         bad_idxs.append(idx)
    # for i in sorted(bad_idxs, reverse=True):
    #     ds[i] = ds[i-3]

    n_train = round(len(ds) * 0.8)
    n_val = round(len(ds) * 0.1)
    ds.x = ds.x.to(torch.float32)
    ds.data.x = ds.data.x.to(torch.float32)

    train_dataset = ds[:n_train]
    val_dataset = ds[n_train:n_train + n_val]
    test_dataset = ds[n_train + n_val:]

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dl, val_dl, test_dl, num_features, num_classes


class FormatAsFloat:
    def __call__(self, data):
        data.x = data.x.to(torch.float32)
        return data

def getQM7b(batch_size):
    init_dataset = QM7b("data/qm7b", transform=torch_geometric.transforms.Compose([FormatAsFloat()]))

    n_train = round(len(init_dataset) * 0.8)
    n_val = round(len(init_dataset) * 0.1)
    train_dataset = init_dataset[:n_train]
    val_dataset = init_dataset[n_train:n_train + n_val]
    test_dataset = init_dataset[n_train + n_val:]

    train_dataset.y = train_dataset.y[:, 0]
    val_dataset.y = val_dataset.y[:, 0]
    test_dataset.y = test_dataset.y[:, 0]

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dl, val_dl, test_dl, num_features, num_classes

def getZINC(batch_size):
    train_dataset = ZINC("data/zinc", subset=True, split='train')
    val_dataset = ZINC("data/zinc", subset=True, split='val')
    test_dataset = ZINC("data/zinc", subset=True, split='test')

    num_features = train_dataset.num_features
    num_classes = None

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dl, val_dl, test_dl, num_features, num_classes


def getPPI(batch_size):
    train_dataset = PPI("data/ppi", split='train')
    val_dataset = PPI("data/ppi", split='val')
    test_dataset = PPI("data/ppi", split='test')

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes

    # train_dataset.data.y = torch.nn.functional.one_hot(train_dataset.data.y[:, 0].to(torch.long), num_classes)
    # val_dataset.data.y = torch.nn.functional.one_hot(val_dataset.data.y[:, 0].to(torch.long), num_classes)
    # test_dataset.data.y = torch.nn.functional.one_hot(test_dataset.data.y[:, 0].to(torch.long), num_classes)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dl, val_dl, test_dl, num_features, num_classes

def getMNIST(batch_size):
    transform = Cartesian(cat=False)
    train_dataset = MNISTSuperpixels("data/mnist", train=True, transform=transform)
    val_percent = 0.1
    val_start = round(len(train_dataset) * (1 - val_percent))

    val_dataset = train_dataset[val_start:]
    train_dataset = train_dataset[:val_start]
    test_dataset = MNISTSuperpixels("data/mnist", train=False)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes

    return train_dl, val_dl, test_dl, num_features, num_classes