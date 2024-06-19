import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from .interrogator import Interrogator
from .interrogators import interrogators


def shrink_and_paste_on_blank(current_image:Image.Image, mask_width:int=64):
    height, width = current_image.height, current_image.width

    prev_image = current_image.resize((height-2*mask_width, width-2*mask_width))
    prev_image = np.array(prev_image.convert("RGBA"))

    blank_image = np.array(current_image.convert("RGBA")) * 0
    blank_image[:, :, 3] = 1
    blank_image[mask_width:height-mask_width, mask_width:width-mask_width, :] = prev_image

    return Image.fromarray(blank_image)


def outpaint_sd_overall(image, model_path):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pipe =  StableDiffusionInpaintPipeline.from_pretrained(model_path)
    pipe.to("cuda")
    interrogator = interrogators['wd14-convnextv2.v1']
    
    new_size = (512, 512)
    image = shrink_and_paste_on_blank([image.resize(new_size)][0])

    mask_image = np.array(image)[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    image = image.convert("RGB")

    result = interrogator.interrogate(image)
    tags = Interrogator.postprocess_tags(result[1], threshold=0.5, escape_tag=True, replace_underscore=True)
    output_image = pipe(prompt=', '.join(tags.keys()), image=image, mask_image=mask_image, num_inference_steps=20).images[0]
    output_image = np.array(output_image)

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)