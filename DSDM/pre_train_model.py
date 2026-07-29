import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from misc.augment import DiffAug
from misc import utils



def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10


    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand

def train_pretrained_models(args):
    """Train the guide pool with the original DSDM pre-training procedure."""
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    save_dir = Path(args.save_pretrain_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    aug, aug_rand = diffaug(args)
    paths = []
    for it in range(args.pretrained_model_number):
    
        model = define_model(args, nclass).to('cuda')
        optim_net = optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        for _ in range(args.pretrained_epochs):
            train_epoch(args,
                        loader_real,
                        model,
                        criterion,
                        optim_net,
                        aug=aug_rand,
                        mixup=args.mixup_net)
        model = model.to('cpu')
        path = save_dir / f'{args.dataset}_model_{it}.pth'
        torch.save(model.state_dict(), path)
        paths.append(path)

    # Keep the loader alive for exactly the same lifetime as the original script.
    del val_loader
    return paths


def train_pretrained_trajectory(args, snapshot_epochs):
    """Train one DSDM guide trajectory and expose its epoch snapshots as a pool."""
    import torch.backends.cudnn as cudnn

    epochs = sorted({int(value) for value in snapshot_epochs})
    max_epochs = int(args.pretrained_epochs)
    if not epochs or epochs[0] <= 0 or epochs[-1] != max_epochs:
        raise ValueError(
            f"trajectory snapshots must end at pretrained_epochs: "
            f"snapshots={epochs} pretrained_epochs={max_epochs}"
        )

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    save_dir = Path(args.save_pretrain_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(
            trainset,
            batch_size=args.batch_real,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    model = define_model(args, trainset.nclass).to('cuda')
    optim_net = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    _, aug_rand = diffaug(args)
    snapshot_to_index = {epoch: index for index, epoch in enumerate(epochs)}
    paths = [save_dir / f'{args.dataset}_model_{index}.pth' for index in range(len(epochs))]

    for epoch in range(1, max_epochs + 1):
        train_epoch(
            args,
            loader_real,
            model,
            criterion,
            optim_net,
            aug=aug_rand,
            mixup=args.mixup_net,
        )
        if epoch in snapshot_to_index:
            index = snapshot_to_index[epoch]
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(state, paths[index])
            print(
                f"[train_guide_trajectory] epoch={epoch}/{max_epochs} "
                f"pool_index={index} path={paths[index]}",
                flush=True,
            )

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"trajectory guide snapshots are incomplete: {missing}")
    del model
    del val_loader
    return paths


if __name__ == '__main__':
    from argument import args

    train_pretrained_models(args)
