import argparse
import logging
import math
import os
import shutil
from datetime import timedelta

import accelerate
import datasets
import torch
from local_accelerate.accelerator import Accelerator
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
from core.utils import flow_viz
from core.utils.utils import backwarp
from core.resample import Sampler
import diffusers
from diffusers import DDPMScheduler
from local_diffusers.models.imagen_unet import SRUnet256
from local_diffusers.models.raft_unet import RAFT_Unet
from local_diffusers.pipelines.DDPM import DDPMPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version
from set_up_dataset import fetch_dataloader
from diffusers.optimization import get_scheduler
import evaluate_diffusers
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.19.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    arr = arr.to(timesteps.device)
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument("--save_images_steps", type=int, default=500, help="How often to save images during training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--max_flow", type=float, default=None, help="exclude extremely large displacements")
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument("--it_aug", action="store_true", help="Whether to use aug from RAFT-it.")
    parser.add_argument("--add_gaussian_noise", action="store_true", help="Whether add gaussian noise to images.")
    parser.add_argument("--filter_epe", action="store_true", help="Whether filter extreme loss value.")
    parser.add_argument("--normalize_range", action="store_true", help="Whether to normalize the flow range into [-1,1].")
    parser.add_argument("--resume_from_model_only", type=str, default=None, help="resume training with model para loaded solely")
    parser.add_argument("--schedule_sampler", type=str, default='normal_left', help="choose the noise distribution")
    parser.add_argument("--Unet_type", type=str, default='SRUnet256', help="determines which UNet used")
    parser.add_argument("--corr_index", type=str, default='noised_flow', help="args for the UNet based on correlation volume")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    if args.Unet_type == 'SRUnet256':
        model_class = SRUnet256
    elif args.Unet_type == 'RAFT_Unet':
        model_class = RAFT_Unet
    else:
        print('error: Unet type undefined!')
        return

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                # make sure to pop weight so that corresponding model is not saved again
                # weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), model_class)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = model_class.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = model_class(channels=8, channels_out=2, sample_size=args.image_size, corr_index=args.corr_index)
    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=model_class,
            model_config=model.config,
        )

    global_step = 0

    if args.resume_from_model_only is not None:
        print('Loading model weights from', args.resume_from_model_only, '/pytorch_model.bin')
        model.load_state_dict(torch.load(args.resume_from_model_only + '/pytorch_model.bin', map_location='cpu'))
        if args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(args.resume_from_model_only, "unet_ema"), model_class)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model
        path = os.path.basename(args.resume_from_model_only)
        global_step = int(path.split("-")[1])

    # Initialize the scheduler
    scale = 1000 / args.ddpm_num_steps
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_start=scale * 0.0001,
        beta_end=scale * 0.02,
        beta_schedule=args.ddpm_beta_schedule,
        clip_sample=False,
        prediction_type=args.prediction_type,
    )

    schedule_sampler = Sampler(args.schedule_sampler, args.ddpm_num_steps)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initializing the dataset
    train_dataloader = fetch_dataloader(args, rank=accelerator.process_index, world_size=accelerator.num_processes)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.num_steps
    )

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return [total_num, trainable_num]

    param_info = get_parameter_number(model)
    accelerator.print(f'########## Total:{param_info[0] / 1e6}M, Trainable:{param_info[1] / 1e6}M ##################')

    # Prepare everything with our `accelerator`.
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)  # train_dataloader

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_steps

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataloader)*args.train_batch_size*accelerator.num_processes}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            # resume_global_step = global_step * args.gradient_accumulation_steps
            # resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    if args.resume_from_model_only or args.resume_from_checkpoint:
        first_epoch = global_step // num_update_steps_per_epoch
        for i in range(global_step):
            lr_scheduler.step()

    # Train!
    should_keep_training = True
    epoch = first_epoch - 1
    while should_keep_training:
        epoch += 1
        # Skip the partial epoch
        if args.resume_from_checkpoint and epoch == first_epoch:
            epoch += 1
        train_dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            model.train()

            for k in batch:
                if type(batch[k]) == torch.Tensor:
                    batch[k] = batch[k].to(model.device)
            batch['flow'] = batch["target"].clone()
            bsz, _, h, w = batch["target"].shape
            if args.normalize_range:
                batch["target"] = torch.clamp(batch["target"] * torch.tensor([1 / w, 1 / h]).view(1, 2, 1, 1).to(model.device), -1, 1)
            # Sample noise that we'll add to the images
            noise = torch.randn(batch["target"].shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)).to(batch["target"].device)
            # Sample a random timestep for each image
            # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=batch["target"].device).long()
            timesteps, weights = schedule_sampler.sample(bsz, batch["target"].device)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_target = noise_scheduler.add_noise(batch["target"], noise, timesteps)
            inputs = torch.cat([2 * (batch["image0"] / 255.0) - 1.0, 2 * (batch["image1"] / 255.0) - 1.0, noisy_target], dim=1)
            if args.max_flow is None:
                valid = (batch['valid'] >= 0.5)
            else:
                mag = torch.sum(batch["flow"] ** 2, dim=1).sqrt()
                valid = (batch['valid'] >= 0.5) & (mag < args.max_flow)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(inputs.to(model.dtype), timesteps, normalize=args.normalize_range).sample
                if args.prediction_type == "epsilon":
                    loss = valid[:, None] * (model_output - noise).abs()  # this could have different weights!
                    metrics = {'loss': loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                elif args.prediction_type == "sample":
                    # alpha_t = _extract_into_tensor(noise_scheduler.alphas_cumprod, timesteps, (batch["target"].shape[0], 1, 1, 1))
                    # snr_weights = alpha_t / (1 - alpha_t)
                    # TODO
                    # may need valid to select pixel points
                    # loss = snr_weights * valid[:, None] * (model_output - batch["target"]).abs()  # use SNR weighting from distillation paper
                    loss = (model_output - batch["target"]).abs()
                    if args.filter_epe:
                        loss_mag = torch.sum(loss ** 2, dim=1).sqrt()
                        mask = loss_mag > 1000
                        if torch.any(mask):
                            logger.info("[Found extrem epe. Filtered out. Max is {}. Ratio is {}]".format(torch.max(loss_mag),
                                                                                                    torch.mean(
                                                                                                        mask.float())))
                            valid = valid & (~mask)
                    loss = weights.view(bsz, 1, 1, 1) * valid[:, None] * loss
                    mask = torch.isnan(loss)
                    if torch.any(mask):
                        logger.info("[Found NAN. Filtered out. Ratio is {}]".format(torch.mean(mask.float())))
                        loss = torch.where(mask, torch.full_like(loss, 0), loss) * (~mask)
                    # metrics
                    if args.normalize_range:
                        model_output_flow = model_output * torch.tensor([w, h]).view(1, 2, 1, 1).to(model.device)
                    epe = torch.sum((model_output_flow - batch["flow"]) ** 2, dim=1).sqrt()
                    epe = epe.view(-1)[valid.view(-1)]
                    metrics = {'loss': loss.mean().item(), 'epe': epe.mean().item(),
                               '1px': (epe < 1).float().mean().item(), '3px': (epe < 3).float().mean().item(),
                               '5px': (epe < 5).float().mean().item(), "lr": lr_scheduler.get_last_lr()[0],
                               "step": global_step}
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                if global_step > args.num_steps:
                    should_keep_training = False
                # Generate sample images for visual inspection
                if global_step % args.save_images_steps == 0:
                    if accelerator.is_main_process:
                        # visualize training
                        denoised_flo = flow_viz.flow_to_image(model_output[0].float().detach().cpu().permute(1, 2, 0).numpy())
                        noised_flow = flow_viz.flow_to_image(noisy_target[0].float().cpu().permute(1, 2, 0).numpy())

                        unet = accelerator.unwrap_model(model)
                        if args.use_ema:
                            ema_model.store(unet.parameters())
                            ema_model.copy_to(unet.parameters())

                        pipeline = DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler
                        )

                        # run pipeline in inference (sample random noise and denoise)
                        images = pipeline(
                            inputs=inputs[0].unsqueeze(0),  # just sample one example
                            batch_size=1,
                            num_inference_steps=args.ddpm_num_steps,
                            output_type="tensor",
                            normalize=args.normalize_range
                        ).images
                        # for visualization
                        flo = flow_viz.flow_to_image(images.float().cpu().permute(0, 2, 3, 1).numpy()[0])
                        flo_gt = flow_viz.flow_to_image((batch["flow"]*batch['valid'][:, None])[0].float().cpu().permute(1, 2, 0).numpy())

                        gt_warpimg1 = backwarp(batch['image1'][0].unsqueeze(0), batch['flow'][0].unsqueeze(0))
                        gt_warpimg1 = gt_warpimg1[0] * batch['valid'][:, None][0]

                        pre_warpimg1 = backwarp(batch['image1'][0].unsqueeze(0), images)[0]
                        pre_valid_warpimg1 = pre_warpimg1 * batch['valid'][:, None][0]

                        gt_img = torch.cat([batch["image0"][0], batch["image1"][0]], dim=-1)
                        warp_img1 = torch.cat([gt_warpimg1, pre_valid_warpimg1], dim=-1)
                        save_img = torch.cat([gt_img, warp_img1], dim=-2)

                        if args.use_ema:
                            ema_model.restore(unet.parameters())

                        # denormalize the images and save to tensorboard
                        if is_accelerate_version(">=", "0.17.0.dev0"):
                            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                        else:
                            tracker = accelerator.get_tracker("tensorboard")
                        tracker.add_images('visualize_training/denoised_flow', denoised_flo, global_step, dataformats='HWC')
                        tracker.add_images('visualize_training/noised_flow', noised_flow, global_step,
                                           dataformats='HWC')
                        tracker.add_images('train_samples/flo_pre', flo, global_step, dataformats='HWC')
                        tracker.add_images('train_samples/flo_gt', flo_gt, global_step, dataformats='HWC')
                        tracker.add_images('train_samples/concat_img', save_img / 255, global_step, dataformats='CHW')
                        tracker.add_images('train_samples/warp_by_pre', pre_warpimg1 / 255, global_step, dataformats='CHW')

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if accelerator.is_main_process:
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if accelerator.is_main_process:
                        # save the model
                        unet = accelerator.unwrap_model(model)

                        if args.use_ema:
                            ema_model.store(unet.parameters())
                            ema_model.copy_to(unet.parameters())

                        pipeline = DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler,
                        )

                        pipeline.save_pretrained(os.path.join(args.output_dir, f"pipeline-{global_step}"))

                        # validate on test set
                        metrics.update(evaluate_diffusers.validate_kitti(pipeline, args=args))

                        if args.use_ema:
                            ema_model.restore(unet.parameters())
            if args.use_ema:
                metrics["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**metrics)
            accelerator.log(metrics, step=global_step)
        progress_bar.close()
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
