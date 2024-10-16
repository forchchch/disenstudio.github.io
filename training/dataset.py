from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from PIL import Image
import random
import torch

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class Region_MixDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        mix_prompt,
        mix_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        mask_root = None
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.instance_data_root = mix_data_root
        
        if not os.path.exists(self.instance_data_root):
            raise ValueError("Instance images root doesn't exists.")
        print("The mixed training data root is: ",self.instance_data_root)
        self.instance_images_path = list(Path(self.instance_data_root).iterdir())
        self.instance_images_path.sort()
        if mask_root is not None:
            self.mask_image_path = list(Path(mask_root).iterdir())
            self.mask_image_path.sort()
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = mix_prompt
        print("The mixed training prompt is: ", self.instance_prompt)
        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        mask_image = Image.open(self.mask_image_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            mask_image = mask_image.convert("RGB")
            
        example["instance_images"] = self.image_transforms(instance_image)
        example["mask_images"] = self.mask_transforms(mask_image)
        example["input_prompt"] = self.instance_prompt

        return example


class Region_MixPriorDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        mix_prompt,
        mix_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        mask_root = None
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.instance_data_root = mix_data_root
        
        if not os.path.exists(self.instance_data_root):
            raise ValueError("Instance images root doesn't exists.")
        print("The mixed training data root is: ",self.instance_data_root)
        self.instance_images_path = list(Path(self.instance_data_root).iterdir())
        self.instance_images_path.sort()
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = mix_prompt
        print("The mixed training prompt is: ", self.instance_prompt)
        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        example["instance_images"] = self.image_transforms(instance_image)
        example["mask_images"] = torch.ones_like(example["instance_images"])
        example["input_prompt"] = self.instance_prompt

        return example