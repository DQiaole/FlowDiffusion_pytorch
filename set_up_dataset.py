from torch.utils.data import (
    DataLoader,
    ConcatDataset
)
from core.datasets_return_dict import AutoFlow
from torch.utils.data.distributed import DistributedSampler


def fetch_dataloader(args, rank=0, world_size=1):
    if args.stage == 'autoflow':
        aug_params = {'crop_size': args.image_size, 'min_scale': 0.2, 'max_scale': 1.0, 'do_flip': True,
                      'add_gaussian_noise': args.add_gaussian_noise}
        autoflow = AutoFlow(aug_params, it_aug=args.it_aug, n_sample=40000)
        train_dataset = ConcatDataset([autoflow, ])
    else:
        raise NotImplementedError(args.stage)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, shuffle=False,
                              num_workers=args.dataloader_num_workers, sampler=train_sampler)
    return train_loader

