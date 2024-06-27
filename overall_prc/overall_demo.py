import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

import config
from .interrogator import Interrogator


def poisson_blend(source, destination, mask, center):
    blended_image = cv2.seamlessClone(source, destination, mask, center, cv2.NORMAL_CLONE)
    return blended_image

def shrink_and_paste_on_blank(current_image: Image.Image, mask_width: int):
    height, width = current_image.height, current_image.width

    prev_image = current_image.resize((width - 2 * mask_width, height - 2 * mask_width))  # Fixed the order
    prev_image = np.array(prev_image.convert('RGBA'))

    blank_image = np.array(current_image.convert('RGBA')) * 0
    blank_image[:, :, 3] = 1
    blank_image[mask_width:height - mask_width, mask_width:width - mask_width, :] = prev_image

    return Image.fromarray(blank_image)

def shrink_and_add_border_from_original(current_image: Image.Image, mask_width: int):
    height, width = current_image.height, current_image.width

    shrinked_image = current_image.resize((width - 2 * mask_width, height - 2 * mask_width))  # Fixed the order
    shrinked_image = shrinked_image.convert("RGBA")
    shrinked_image_array = np.array(shrinked_image)

    result_image_array = np.array(current_image.convert("RGBA"))
    result_image_array[mask_width:height - mask_width, mask_width:width - mask_width, :] = shrinked_image_array

    return Image.fromarray(result_image_array)

def shrink_and_add_blurred_border(current_image: Image.Image, mask_width: int):
    height, width = current_image.height, current_image.width

    shrinked_image = current_image.resize((width - 2 * mask_width, height - 2 * mask_width))  # Fixed the order
    shrinked_image = shrinked_image.convert("RGBA")
    shrinked_image_array = np.array(shrinked_image)

    result_image_array = np.array(current_image.convert("RGBA"))
    blurred_image = current_image.filter(ImageFilter.GaussianBlur(radius=24))
    blurred_image_array = np.array(blurred_image.convert("RGBA"))

    result_image_array[:, :] = blurred_image_array
    result_image_array[mask_width:height - mask_width, mask_width:width - mask_width, :] = shrinked_image_array

    return Image.fromarray(result_image_array)

def overall_prc(image, aimodels, device='cuda:2'):
    
    pipe, interrogator, sr_model, sod_model = aimodels.pipe1, aimodels.sd_tagger, aimodels.sr2_1_model, aimodels.sod_model
    to_tensor, to_pil = T.ToTensor(), T.ToPILImage()
    image_original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = image_original.size
    
    new_size = (width, height)
    while width + height > 2048:
        width = width // 2
        height = height // 2
        new_size = (width, height)
    
    MASK_WIDTH_PERCENTAGE = 0.15
    MASK_WIDTH = int(height * MASK_WIDTH_PERCENTAGE)

    image_original = image_original.resize(new_size)
    image = shrink_and_add_border_from_original(image_original, MASK_WIDTH)
    image = image.convert('RGB')

    mask_image = np.array(shrink_and_paste_on_blank(image_original, MASK_WIDTH))[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert('L')  # Convert to single channel

    result = interrogator.interrogate(image_original)
    tags = Interrogator.postprocess_tags(result[1], threshold=0.75, escape_tag=True, replace_underscore=True)
    for i in tags: print(f'{i} : {tags[i]}')

    while True:
        output = pipe(prompt=', '.join(tags.keys()), negative_prompt=config.negative_prompt, image=image, mask_image=mask_image, num_inference_steps=20, strength=0.94)
        if not output.nsfw_content_detected[0]:
            output_image = output.images[0]
            break
    output_image = output_image.resize(new_size)
    
    in_ = np.array(output_image, dtype=np.float32) 
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_2, _ = in_.transpose((2,0,1)), tuple(in_.shape[:2])
    in_2 = torch.Tensor(in_2).unsqueeze(0)

    with torch.no_grad():
        preds = sod_model(in_2, mode=3)
        pred_sal = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
        pred_sal = 255 * pred_sal
        pred_sal = np.where(pred_sal > 126, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(pred_sal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        max_val = float('inf')*-1
        m_bbx = 0
        for idx, (x, y, w, h) in enumerate(bounding_boxes):
            if w * h > max_val:
                max_val = w * h
                m_bbx = idx
            
    x, y, w, h = bounding_boxes[m_bbx]
    x_point = x + w//2

    ### 9:16
    crop_width = int(height*0.5625)
    crop_height = height

    y, w, h = 0, crop_width, height
    h = height

    if x_point - crop_width//2 < 0: x = 0
    elif x_point + crop_width//2 > width: x = width - crop_width
    else: x = x_point - crop_width//2

    t_width, t_height = image_original.size
    shrinked_image = image_original.resize((t_width - 2 * MASK_WIDTH, t_height - 2 * MASK_WIDTH))
    shrinked_image = shrinked_image.convert("RGBA")

    shrink_black_image = shrink_and_paste_on_blank(image_original, MASK_WIDTH)
    mask_image = Image.fromarray(np.array(shrink_black_image)[:, :, 3]).convert('L')

    shrinked_image_np = np.array(shrink_black_image.convert("RGB"))
    mask_image_np = np.array(mask_image)
    
    output_image_np = np.array(output_image.convert("RGB"))
    center = (output_image_np.shape[1] // 2, output_image_np.shape[0] // 2)
    blended_image = Image.fromarray(poisson_blend(shrinked_image_np, output_image_np, mask_image_np, center))

    cropped_image = blended_image.crop((x, y, x + w, y + h))
    cropped_image_tensor = to_tensor(cropped_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_result = sr_model(cropped_image_tensor).clamp(0, 1)
    sr_image = to_pil(sr_result.squeeze(0).cpu())
    final_image = sr_image.resize((1080, 1838))

    return cv2.cvtColor(np.array(final_image), cv2.COLOR_BGR2RGB)