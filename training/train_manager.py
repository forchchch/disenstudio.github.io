import os
import subprocess

training_ids = [i for i in [0]]

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
###you can also use the following checkpoint for pretraining
#MODEL_NAME="emilianJR/epiCRealism"
OUTPUT_DIR="./training_results"
INSTANCE_DIR="../data/instance"
MASK_DIR="../data/instance_mask"
MIX_PATH="../preprocess/mix_data"
MIX_MASK_PATH="../preprocess/mix_masks"
MIX_B_PATH="../preprocess/mix_b_data"
CLASS_DIR="../preprocess/mix_class_data"



def trial(args):
    cmd = "CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision='fp16' region_aware_train.py "
    cmd += " --train_text_encoder"
    cmd += " --use_background"
    # cmd += " --mix_loss_mask"

    for k in args:
        v = args[k]
        cmd += ' --' + k
        if type(v) == str:
            cmd += f" {v}"
        elif type(v) == int:
            cmd += f" {int(v)}"
        elif type(v) == float:
            cmd += f" {float(v)}"
        else:
            raise NotImplementedError(f"not support key {k} with a {type(v)} value {v}")
    print(cmd)
    try:
        subprocess.run(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"failed ...\n{cmd}")

if __name__ == "__main__":
    for dataset_id in training_ids:
        args = {
            "pretrained_model_name_or_path": MODEL_NAME,
            "benchmark_id": dataset_id,
            "dataloader_num_workers":1,
            "instance_data_dir": INSTANCE_DIR, 
            "mask_data_dir":MASK_DIR,
            "mix_dir":MIX_PATH,
            "mix_b_dir":MIX_B_PATH,
            "mix_mask_dir":MIX_MASK_PATH,
            "class_data_dir":CLASS_DIR,
            "exp_name":"DisenStudio",
            "output_dir": OUTPUT_DIR,
            "resolution": 512,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "lr_scheduler": "cosine",
            "max_grad_norm":1,
            "seed":1337,
            "checkpointing_steps": 100,
            "lr_warmup_steps": 0,
            "max_train_steps": 2000,
            "rank":16,
            "mix_weight":1.0,
            "single_weight":1.0,
            "prior_weight":1.0,
            "text_lr":2e-5
        }
        trial(args)