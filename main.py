import os
import sys
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
    touch_dtype=torch.bfloat16,
    text_encoder_3=None,
    tokenizer_3=None
)
pipe.to(device=f"cuda:{device_with_most_processors}")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]

os.makedirs("./images", exist_ok=True)
image.save("./images/capybara.png")
