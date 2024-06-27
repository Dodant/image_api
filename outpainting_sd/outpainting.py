import cv2
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

import config
from .interrogator import Interrogator
from .interrogators import interrogators


def shrink_and_paste_on_blank(current_image:Image.Image, mask_width:int=64):
    height, width = current_image.height, current_image.width

    prev_image = current_image.resize((height-2*mask_width, width-2*mask_width))
    prev_image = np.array(prev_image.convert('RGBA'))

    blank_image = np.array(current_image.convert('RGBA')) * 0
    blank_image[:, :, 3] = 1
    blank_image[mask_width:height-mask_width, mask_width:width-mask_width, :] = prev_image

    return Image.fromarray(blank_image)


def shrink_and_add_border_from_original(current_image: Image.Image, mask_width: int=64):
    height, width = current_image.height, current_image.width

    shrinked_image = current_image.resize((width - 2 * mask_width, height - 2 * mask_width))
    shrinked_image = shrinked_image.convert("RGBA")
    shrinked_image_array = np.array(shrinked_image)

    result_image_array = np.array(current_image.convert("RGBA"))
    result_image_array[mask_width : height - mask_width, mask_width : width - mask_width, :] = shrinked_image_array
    
    return Image.fromarray(result_image_array)


def shrink_and_add_blurred_border(current_image: Image.Image, mask_width: int=64):
    height, width = current_image.height, current_image.width

    shrinked_image = current_image.resize((width - 2 * mask_width, height - 2 * mask_width))
    shrinked_image = shrinked_image.convert("RGBA")
    shrinked_image_array = np.array(shrinked_image)

    result_image_array = np.array(current_image.convert("RGBA"))
    blurred_image = current_image.filter(ImageFilter.GaussianBlur(radius=24))
    blurred_image_array = np.array(blurred_image.convert("RGBA"))

    result_image_array[:, :] = blurred_image_array
    result_image_array[mask_width : height - mask_width, mask_width : width - mask_width, :] = shrinked_image_array

    return Image.fromarray(result_image_array)


def outpaint_sd_overall(image, pipe, interrogator):
    NEGATIVE_PROMPT = 'text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality, ((nsfw))'
    
    image_original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    new_size = (512, 512)
    image_original_ =[image_original.resize(new_size)][0]
    image = shrink_and_add_blurred_border(image_original_)
    image = image.convert('RGB')
    
    mask_image = np.array(shrink_and_paste_on_blank(image_original_))[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert('RGB')

    result = interrogator.interrogate(image_original)
    tags = Interrogator.postprocess_tags(result[1], threshold=0.75, escape_tag=True, replace_underscore=True)
    for i in tags: print(f'{i} : {tags[i]}')

    for _ in range(3):
        output = pipe(prompt=', '.join(tags.keys()), negative_prompt=NEGATIVE_PROMPT, image=image, mask_image=mask_image, num_inference_steps=20, strength=0.925)
        if not output.nsfw_content_detected[0]:
            output_image = output.images[0]
            break
    output_image = np.array(output_image)

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)