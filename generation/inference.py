import torch
from diffusers import DDIMScheduler, MotionAdapter
from region_animatediff_pipeline import Region_AnimateDiffPipeline
from diffusers.utils import export_to_gif
import os
from diffusers import UNet2DConditionModel, UNetMotionModel


adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
#model_id = "emilianJR/epiCRealism"
### or you can use the following parameters
model_id = "runwayml/stable-diffusion-v1-5"
pipe = Region_AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
lora_path = "../training/training_results/DisenStudio/catandcat2/checkpoint/checkpoint-700"
pipe.load_lora_weights(lora_path)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
    )
pipe.scheduler = scheduler

# # enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
pic_width = 512
pic_height = 512
context_prompt = "beside a river"
nega_prompt = "bad quality; worst quality; low resolution"
seed = 5


torch.manual_seed(seed)

###first object sks, second object krk, third object slk, fourth object ofk
region_prompt_list = ["a sks cat playing the guitar beside a river", "a krk cat wearing a yellow scarf beside a river" ]
region_box_list = [  [30, 0, 482, 220 ], [30, 256, 482, 512 ] ]
region_num = len(region_prompt_list)
for j in range(region_num):
    region_box = region_box_list[j]
    region_box_list[j][0], region_box_list[j][2] = region_box_list[j][0]/pic_height, region_box_list[j][2]/pic_height
    region_box_list[j][1], region_box_list[j][3] = region_box_list[j][1]/pic_width, region_box_list[j][3]/pic_width


region_list = [ ( region_prompt_list[i], nega_prompt, region_box_list[i] ) for i in range(region_num) ]
input_prompt = [ ( context_prompt, region_list ) ]  

output = pipe(
    prompt= input_prompt ,
    negative_prompt="bad quality; worst quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(seed),
)
frames = output.frames[0]
export_to_gif(frames, "./video"+ str(seed)+".gif")