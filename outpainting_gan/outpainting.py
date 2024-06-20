import skimage
import numpy as np
import tensorflow as tf
from skimage import transform
from scipy.ndimage.morphology import distance_transform_edt

input_size = 128
output_size = 192
expand_size = (output_size - input_size) // 2
patch_w = output_size // 8
patch_h = output_size // 8
patch = (1, patch_h, patch_w)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def resize_masking(input_img, size):
    resized = transform.resize(input_img, size, anti_aliasing=True)
    masked_img = np.ones((output_size, output_size, 3))
    masked_img[expand_size:-expand_size, expand_size:-expand_size, :] = resized
    assert(masked_img.shape[0] == output_size)
    assert(masked_img.shape[1] == output_size)
    assert(masked_img.shape[2] == 3)
    masked_img = masked_img.transpose(2, 0, 1)
    masked_img = tf.expand_dims(masked_img, axis=0)
    return masked_img

def preprocess_image_for_int8(input_img, quantization_params=None, to_int=False):
    input_img = input_img / quantization_params['scale'] + quantization_params['zero_point']
    input_img = np.round(input_img).astype(np.int8)
    return input_img

def predict_image(interpreter, input_img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_img = np.array(input_img, dtype=np.float32) if input_details[0]['dtype'] == np.float32 else np.array(input_img, dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output_img = interpreter.get_tensor(output_details[0]['index'])
    return output_img

def postprocess_image(image, quantization_params=None, to_int=False):
    image = image.squeeze().transpose(1, 2, 0)
    if to_int:
        image = (image - quantization_params['zero_point']) * quantization_params['scale']
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.squeeze().transpose(1, 2, 0)
        image = np.clip(image * 255, 0, 255)
        image = image.astype(np.uint8)
    else:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8) 
    return image

def blend(output_img, input_img, blend_width=8):
    in_factor = input_size / output_size
    if input_img.shape[1] < in_factor * output_img.shape[1]:
        out_width, out_height = output_img.shape[1], output_img.shape[0]
        in_width, in_height = int(out_width * in_factor), int(out_height * in_factor)
        input_img = transform.resize(input_img, (in_height, in_width), anti_aliasing=True)
    else:
        in_width, in_height = input_img.shape[1], input_img.shape[0]
        out_width, out_height = int(in_width / in_factor), int(in_height / in_factor)
        output_img = transform.resize(output_img, (out_height, out_width), anti_aliasing=True)

    src_mask = np.zeros((output_size, output_size))
    src_mask[expand_size+1:-expand_size-1, expand_size+1:-expand_size-1] = 1 # 1 extra pixel for safety
    src_mask = distance_transform_edt(src_mask) / blend_width
    src_mask = np.minimum(src_mask, 1)
    src_mask = transform.resize(src_mask, (out_height, out_width), anti_aliasing=True)
    src_mask = np.tile(src_mask[:, :, np.newaxis], (1, 1, 3))
    
    input_pad = np.zeros((out_height, out_width, 3))
    x1 = (out_width - in_width) // 2
    y1 = (out_height - in_height) // 2
    input_pad[y1:y1+in_height, x1:x1+in_width, :] = input_img
    
    blended = input_pad * src_mask + output_img * (1 - src_mask)

    return blended, src_mask

def outpaint(output_img, input_img):
    norm_input_img = input_img.copy().astype('float')
    if np.max(norm_input_img) > 1: norm_input_img /= 255
    blended_img, _ = blend(output_img, norm_input_img)
    blended_img = np.clip(blended_img, 0, 1)

    return output_img, blended_img