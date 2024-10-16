import argparse
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from utils import my_make_dir, get_logger, gen_mix_image, mix_background, make_prior_prompt
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from dataset import Region_MixDataset, Region_MixPriorDataset
from region_pipeline_lora import RegionCrossFrameAttnProcessor,RegionVideoPipeline_lora
import random
import torch.nn as nn
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
# import open_clip
import json
import cv2
import imageio
import copy


def create_region_mask(input_size=64, index=0, total_num=2):
    mask = torch.zeros(1,1,64,input_size)
    mask[:, :, :, int((index*input_size+0.0)/total_num): int(((index+1)*input_size+0.0 )/total_num) ] = 1.0
    return mask

def make_instance_prompt(region_prompt_list, region_box_list, context_prompt="",width=512, height=512):
    region_box_list1 = copy.deepcopy(region_box_list)
    region_num = len(region_prompt_list)
    for j in range(region_num):
        region_box = region_box_list1[j]
        region_box_list1[j][0], region_box_list1[j][2] = region_box_list1[j][0]/height, region_box_list1[j][2]/height
        region_box_list1[j][1], region_box_list1[j][3] = region_box_list1[j][1]/width, region_box_list1[j][3]/width

    region_list = [ ( region_prompt_list[i], region_box_list1[i] ) for i in range(region_num) ]
    input_prompt =  ( context_prompt, region_list )   
    
    return input_prompt

def obtain_separate_prompt(mixed_prompt):
    context_prompt, region_prompt = mixed_prompt[0][0], mixed_prompt[0][1]
    region_num = len(region_prompt)
    separate_prompts = []
    for i in range(region_num):
        prompt = [ (context_prompt, [region_prompt[i]]) ]
        separate_prompts.append( prompt )
    return separate_prompts        

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mix_dir",
        type=str,
        default=None,
        required=True,
        help="root for mix data",
    )
    parser.add_argument(
        "--mix_b_dir",
        type=str,
        default=None,
        required=True,
        help="root for mix data",
    )
    parser.add_argument(
        "--mix_mask_dir",
        type=str,
        default=None,
        required=True,
        help="root for mix masks",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="used to identify the name of the experiment",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--mask_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of mask images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--benchmark_id",
        type=int,
        default=0,
        required=True,
        help="the id of the used benchmark",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--prior_weight",
        type=float,
        default=1.0,
        help=("how much we disentangle."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )

    parser.add_argument(
        "--use_background",
        default=False,
        action="store_true",
        help="Flag to use background.",
    )
    parser.add_argument(
        "--mix_loss_mask",
        default=False,
        action="store_true",
        help="Flag to add mask to the mixed loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Whether to use the curriculum training.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--mix_weight",
        type=float,
        default=1.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--single_weight",
        type=float,
        default=1.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--text_lr",
        type=float,
        default=1e-5,
        help="learning rate for text encoder.",
    )
    parser.add_argument(
        "--img_lr",
        type=float,
        default=1e-4,
        help="learning rate for image encoder.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    pixel_masks = [example["mask_images"] for example in examples]


    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    pixel_masks = torch.stack(pixel_masks)
    pixel_masks = pixel_masks.to(memory_format=torch.contiguous_format).float()

    input_prompt = [example["input_prompt"] for example in examples]

    batch = {
        "input_prompt": input_prompt,
        "pixel_values": pixel_values,
        "pixel_masks": pixel_masks
    }

    return batch


def encode_region_prompt(prompt, text_encoder, tokenizer):

    device = text_encoder.device
    context_prompt, region_list = prompt[0][0], prompt[0][1]
    # print("context_prompt:",context_prompt)
    context_prompt_input_ids = tokenizer(
        context_prompt,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
    ).input_ids
    prompt_embeds = text_encoder(context_prompt_input_ids.to(device), attention_mask=None)[0]
    prompt_embeds.to(dtype=text_encoder.dtype, device=device)    
    prompt_embeds = torch.cat([prompt_embeds])
    
    region_embeds_list = []
    for idx, region in enumerate(region_list):
        region_prompt, pos = region
        region_prompt_input_ids = tokenizer(
            region_prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt').input_ids
        region_embeds = text_encoder(region_prompt_input_ids.to(device), attention_mask=None)[0]
        region_embeds.to(dtype=text_encoder.dtype, device=device)
        region_embeds_list.append( (torch.cat([region_embeds]), pos) )
    return prompt_embeds, region_embeds_list

def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def main(args):
    with open( os.path.join(args.instance_data_dir, "benchmark.json") ) as f:
        benchmark = json.load(f)
    dataset_info = benchmark[args.benchmark_id] 
    name_list = []
    class_list = []
    dataset_info_keys = sorted( list(dataset_info.keys()) )
    for key in dataset_info_keys:
        if key.startswith("ins"):
            name_list.append( dataset_info[key] )
        if key.startswith("cls"):
            class_list.append( dataset_info[key] )
    print(name_list)
    print(class_list)
    special_token_list = ["sks", "krk", "slk", "ofk"]
    subject_num = len(name_list)

    pic_width = 512
    pic_height = 512
    mix_prompt_list = []
    mix_box_list = []
    mix_valid_prompt_list = []
    null_box_list = []
    
    prior_prompt_list = []
    natural_prompt_list = []
    global_prompt = ""
    
    for i in range(subject_num):
        tmp_dict = {}
        single_prompt = "a " + special_token_list[i] + " " + class_list[i]
        global_prompt += single_prompt
        if not i == subject_num-1:
            global_prompt += ", and "

        single_valid_prompt = "a " + special_token_list[i] + " " + class_list[i]
        prior_single_prompt = "a cute and pretty " + class_list[i]
        natural_single_prompt = "a " + class_list[i] 
        mix_prompt_list.append(single_prompt)
        mix_valid_prompt_list.append(single_valid_prompt)
        prior_prompt_list.append(prior_single_prompt)
        natural_prompt_list.append(natural_single_prompt)
        mix_box_list.append( [0, int( (i+0.0)*pic_width/subject_num ) ,pic_height, int( (i+1.0)*pic_width/subject_num )] )
        null_box_list.append( [0, 0, 0, 0] )

    ######used to train the mix images
    mix_prompt = make_instance_prompt(mix_prompt_list, mix_box_list)
    ######used to generate the prior images
    prior_prompt = make_instance_prompt(prior_prompt_list, mix_box_list)
    ######used to train the prior images
    natural_prompt = make_instance_prompt(natural_prompt_list, mix_box_list)
    ######used to verify the generation
    valid_mix_prompt = make_instance_prompt(mix_valid_prompt_list, mix_box_list)
    
    clip_trans = transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    sub_dirs = dataset_info["note"].split(" ")
    sub_dir = ""
    for item in sub_dirs:
        sub_dir += item
    olog_dir, oimg_dir, ocheck_dir = my_make_dir( args.output_dir, os.path.join(args.exp_name, sub_dir)  )
    logger = get_logger(os.path.join(olog_dir, 'logging.txt'), "glogger")
    mix_num = 1
    mix_data_dir = gen_mix_image(args.instance_data_dir, name_list, args.mix_dir, target_size=512*mix_num, use_copy=False)
    mix_mask_dir = gen_mix_image(args.mask_data_dir, name_list, args.mix_mask_dir, target_size=512*mix_num, use_copy=False)
    if args.use_background:
        background_dir = "../data/background"
        mix_data_dir = mix_background(mix_data_dir, mix_mask_dir, background_dir, args.mix_b_dir, 512*mix_num, use_copy=False)
    
    logger.info(f'{mix_prompt}')
    logger.info(f'{prior_prompt}')

    logger.info(f'mix dir:{mix_data_dir}')
    logger.info(f'mix mask dir: {mix_mask_dir}')
    
    
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
        
    logger.info(accelerator.state)
    logger.info(args)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    

    # Generate class images if prior preservation is enabled.
    # Handle the repository creation

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)


    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()


    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        # if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
        #     lora_attn_processor_class = LoRAAttnAddedKVProcessor
        # else:
        #     lora_attn_processor_class = (
        #         LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        #     )
        lora_attn_processor_class = LoRAAttnProcessor
        
        module = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.rank
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        lora_list = [attn_processor.to_q_lora, attn_processor.to_k_lora, attn_processor.to_v_lora, attn_processor.to_out_lora]
        new_module = RegionCrossFrameAttnProcessor(lora_list).to(accelerator.device)
        unet_lora_attn_procs[name] = new_module
    unet.set_attn_processor(unet_lora_attn_procs)
    
    ##############generate mix samples
    gen_pic_height = 512
    gen_pic_width = 512*subject_num
    prior_name = ""
    for i in range(subject_num):
        if i==0:
            prior_name += class_list[i]
        else:
            prior_name += ("_" + class_list[i] )
    class_image_dir = os.path.join(args.class_data_dir, prior_name, "size"+str(512*mix_num) )
    class_images_dir = Path(class_image_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < args.num_class_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        if args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = RegionVideoPipeline_lora.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unet,
                    text_encoder=text_encoder,
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(accelerator.device)
        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")
        for i in range(num_new_images):
            n_prior_prompt = make_prior_prompt(prior_prompt)
            print(i, n_prior_prompt)
            gen_image_mix = pipeline([ n_prior_prompt ], guidance_scale=7.0, video_length=1, t0 = 48 , t1 =48, height=gen_pic_height, width=gen_pic_width).images[0]
            gen_image_mix = (gen_image_mix * 255).astype("uint8")
            save_gen_mix = cv2.resize( gen_image_mix, dsize=(512*mix_num, 512) )
            imageio.imwrite(os.path.join(class_images_dir, str(i+cur_class_images)+".jpg"), save_gen_mix)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
            else:
                torch.save(model.state_dict() ,os.path.join(output_dir,"my_fuser.pt"))
                # raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        LoraLoaderMixin.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        [{"params": itertools.chain(unet_lora_parameters), "lr": args.learning_rate},
         {"params": itertools.chain(text_lora_parameters), "lr": args.text_lr},
        ] if args.train_text_encoder
        else [ {"params": itertools.chain(unet_lora_parameters), "lr": args.learning_rate},
            ]
         )

    optimizer = optimizer_class(
        params_to_optimize,
        # lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    pre_computed_encoder_hidden_states = None
    validation_prompt_encoder_hidden_states = None
    validation_prompt_negative_prompt_embeds = None
    pre_computed_class_prompt_encoder_hidden_states = None


    mix_dataset = Region_MixDataset(
        mix_prompt=mix_prompt,
        mix_data_root=mix_data_dir,
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        mask_root = mix_mask_dir
        
    )

    mix_dataloader = torch.utils.data.DataLoader(
        mix_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
    ) 

    train_dataset = Region_MixPriorDataset(
        mix_prompt=natural_prompt,
        mix_data_root=class_image_dir,
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        mask_root = mix_mask_dir
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
    ) 



    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, lr_scheduler = accelerator.prepare(unet, text_encoder, optimizer, lr_scheduler)
        mix_dataloader = accelerator.prepare(mix_dataloader)
        train_dataloader = accelerator.prepare(train_dataloader)
    else:
        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
        train_dataloader = accelerator.prepare(train_dataloader)
        mix_dataloader = accelerator.prepare(mix_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main proce
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("multi-subject-lora", config=tracker_config)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** start Running training *****")

    global_step = 0
    first_epoch = 0
    individual_weight = 1.0
    multi_weight = 1.0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
   
    mixiter = iter(mix_dataloader)
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            with accelerator.accumulate(unet):
                        
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                mask_values = batch["pixel_masks"].to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                timesteps = timesteps.long()
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                #dimension [1,4,64,64]
                # Get the text embedding for conditioning
                prompt_embeds, region_list = encode_region_prompt(batch["input_prompt"], text_encoder, tokenizer)
                region_cross_attention_kwargs = {
                    'region_list':region_list,
                    'height':pic_height,
                    'width':pic_width*mix_num
                }
                #dimension [1,77,1024]
                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                model_pred = unet(noisy_model_input, timesteps, prompt_embeds, cross_attention_kwargs=region_cross_attention_kwargs).sample
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = ( F.mse_loss(model_pred.float(), target.float(), reduction="none") ).mean()  
                accelerator.backward(args.prior_weight*loss)
                
                loss_main = 0.0
                try:
                    batch_mix = next(mixiter)
                except:
                    mixiter = iter(mix_dataloader)
                    batch_mix = next(mixiter)
                pixel_values_mix = batch_mix["pixel_values"].to(dtype=weight_dtype)
                mask_values_mix = batch_mix["pixel_masks"].to(dtype=weight_dtype)
                mask_mix = ( ( torch.sum(mask_values_mix, dim=1, keepdim=True) > 0.5 ) + 0.0 ).to(dtype=weight_dtype)


                if vae is not None:
                    # Convert images to latent space
                    model_input_mix = vae.encode(pixel_values_mix).latent_dist.sample()
                    model_input_mix = model_input_mix * vae.config.scaling_factor
                else:
                    model_input_mix = pixel_values_mix

                # Sample noise that we'll add to the latents
                noise_mix = torch.randn_like(model_input_mix)
                bsz, channels, height, width = model_input_mix.shape
                # Sample a random timestep for each image
                timesteps_mix = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input_mix.device)
                timesteps_mix = timesteps_mix.long()
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input_mix = noise_scheduler.add_noise(model_input_mix, noise_mix, timesteps_mix)

                # Get the text embedding for conditioning

                prompt_embeds_mix, region_list_mix = encode_region_prompt(batch_mix["input_prompt"], text_encoder, tokenizer)
                region_cross_attention_kwargs_mix = {
                    'region_list':region_list_mix,
                    'height':pic_height,
                    'width':pic_width*mix_num
                }
                # Predict the noise residual
                model_pred_mix = unet(noisy_model_input_mix, timesteps_mix, prompt_embeds_mix, cross_attention_kwargs=region_cross_attention_kwargs_mix).sample
                ###(1,4,64,64)
                target_mix = noise_mix
                resized_mask_mix = F.interpolate(mask_mix, [64,64*mix_num])
                if args.mix_loss_mask:
                    loss_mix = ( F.mse_loss(model_pred_mix.float(), target_mix.float(), reduction="none")*resized_mask_mix ).mean()
                else:
                    loss_mix = ( F.mse_loss(model_pred_mix.float(), target_mix.float() , reduction="none") ).mean()
                

                loss_main = loss_main + args.mix_weight*loss_mix
                
                separate_prompts = obtain_separate_prompt( [valid_mix_prompt] )
                random_subject_j = random.randint(0, subject_num-1)
                
                prompt_j = separate_prompts[random_subject_j]
                prompt_embeds_j, region_list_j = encode_region_prompt(prompt_j, text_encoder, tokenizer)
                region_cross_attention_kwargs_j = {
                    'region_list':region_list_j,
                    'height':pic_height,
                    'width':pic_width*mix_num
                }
                # Predict the noise residual
                model_pred_j = unet(noisy_model_input_mix, timesteps_mix, prompt_embeds_j, cross_attention_kwargs=region_cross_attention_kwargs_j).sample
                ###(1,4,64,64)
                region_mask_j = create_region_mask(input_size=64*mix_num, index=random_subject_j, total_num=subject_num).to(accelerator.device)
                sample_mask_j = resized_mask_mix * region_mask_j
                target_j = noise_mix
                # import cv2
                # save_mask = (255.0*sample_mask).squeeze(0).cpu().permute(1,2,0).numpy()
                # print(save_mask.shape)
                # cv2.imwrite("./mask_debug.jpg", save_mask)
                # exit(0)
                loss_j = ( F.mse_loss(model_pred_j.float(), target_j.float(), reduction="none")*sample_mask_j ).mean()
                # loss_mix = ( F.mse_loss(model_pred_mix.float(), target_mix.float(), reduction="none") ).mean()
                loss_main = loss_main + args.single_weight*loss_j                        
                
                accelerator.backward(loss_main)

                if accelerator.sync_gradients:
                    params_to_clip = (   itertools.chain(unet_lora_parameters, text_lora_parameters) if args.train_text_encoder 
                                      else itertools.chain(unet_lora_parameters)
                                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(ocheck_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info(f"We are running validation on global step :{global_step}.")
                        pipeline = RegionVideoPipeline_lora.from_pretrained(
                                    args.pretrained_model_name_or_path,
                                    unet=accelerator.unwrap_model(unet),
                                    text_encoder=None if args.pre_compute_text_embeddings else accelerator.unwrap_model(text_encoder),
                                    revision=args.revision,
                                    torch_dtype=weight_dtype,
                                )
                        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        current_img_dir = os.path.join( oimg_dir, "step_"+str(global_step) )
                        os.makedirs( current_img_dir, exist_ok=True)
                        with torch.cuda.amp.autocast():
                            valid_prompts = obtain_separate_prompt( [valid_mix_prompt] )
                            for image_num in range(subject_num):
                                # print("before generation:",concepts_list[image_num]["instance_prompt"])
                                image1 = pipeline( valid_prompts[image_num] , guidance_scale=7.0, video_length=1, t0 = 48 , t1 =48, height=pic_height, width=pic_width*mix_num).images[0]
                                image1 = (image1 * 255).astype("uint8")
                                imageio.imwrite(os.path.join(current_img_dir, "s_" + str(image_num) + ".jpg"), image1)
                            # print("before generation:", [mix_prompt])
                            image_mix = pipeline([ valid_mix_prompt ], guidance_scale=7.0, video_length=1, t0 = 48 , t1 =48, height=pic_height, width=pic_width*mix_num).images[0]
                            image_mix = (image_mix * 255).astype("uint8")
                            imageio.imwrite(os.path.join(current_img_dir, "s_mix.jpg"), image_mix)
                        # print("after generation:", concepts_list[0]["instance_prompt"], concepts_list[1]["instance_prompt"])
                        # print("after generation:",[ mix_prompt ])
                        del pipeline
                        torch.cuda.empty_cache()
                        
            logger.info(f"loss: {loss.detach().item()}, lr: {lr_scheduler.get_last_lr()[0]}")

            if global_step >= args.max_train_steps:
                break

        
                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        if text_encoder is not None and args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder = text_encoder.to(torch.float32)
            text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)
        else:
            text_encoder_lora_layers = None

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
