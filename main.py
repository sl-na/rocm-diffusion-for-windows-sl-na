import os
import sys
from datetime import datetime
import torch
from diffusers import StableDiffusion3Pipeline

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting...")
    sys.exit()

device_with_most_processors = 0
if torch.cuda.device_count() > 1:
    proccessor_count = 0
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).multi_processor_count > proccessor_count:
            device_with_most_processors = i
            proccessor_count = torch.cuda.get_device_properties(i).multi_processor_count


pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    touch_dtype=torch.bfloat16
)
pipe.to(device=f"cuda:{device_with_most_processors}")
pipe.enable_model_cpu_offload(gpu_id=device_with_most_processors)

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
    height=768,
    width=768
).images[0]

os.makedirs("./images", exist_ok=True)
now = datetime.now()
ms = now.microsecond // 1000
now_str = f"{now:%Y-%m%d-%H%M%S}_{ms}"
image.save(f"./images/capybara{now_str}.png")
