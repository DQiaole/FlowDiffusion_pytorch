from core.datasets_return_dict import KITTI
import torch.utils.data as data
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def validate_kitti(pipeline, args=None):
    """ Peform validation using the KITTI-2015 (train) split """
    aug_params = {'crop_size': args.image_size, }
    val_dataset = KITTI(aug_params, split='training', resize_for_test=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.train_batch_size, pin_memory=True, shuffle=False, num_workers=4)

    out_list, epe_list = [], []
    for batch in tqdm(val_loader):
        for k in batch:
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].cuda()

        # run pipeline in inference (sample random noise and denoise)
        inputs = torch.cat([2 * (batch["image0"] / 255.0) - 1.0, 2 * (batch["image1"] / 255.0) - 1.0], dim=1)
        pipeline.unet = pipeline.unet.to(torch.bfloat16)
        images = pipeline(
            inputs=inputs.to(torch.bfloat16),  # just sample one example
            batch_size=inputs.shape[0],
            num_inference_steps=args.ddpm_num_steps,
            output_type="tensor",
            normalize=args.normalize_range
        ).images

        epe = torch.sum((images - batch['target'])**2, dim=1).sqrt()
        mag = torch.sum(batch['target']**2, dim=1).sqrt()
        for index in range(inputs.shape[0]):
            epe_indexed = epe[index].view(-1)
            mag_indexed = mag[index].view(-1)
            val = batch['valid'][index].view(-1) >= 0.5
            out = ((epe_indexed > 3.0) & ((epe_indexed/mag_indexed) > 0.05)).float()
            epe_list.append(epe_indexed[val].mean().cpu().item())
            out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}
