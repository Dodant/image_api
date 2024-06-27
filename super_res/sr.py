from PIL import Image

import cv2
import numpy as np
import tensorflow as tf


def sr_overall(image, interpreter, resolution=128, tf_=True):
    new_size = (resolution, resolution)
    init_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    init_image_np = np.array(init_image.resize(new_size)).astype(np.float32) / 255.0
    init_image_np = np.expand_dims(init_image_np, axis=0)
    if not tf_:
        init_image_np = init_image_np.transpose(0, 3, 1, 2)
    
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], init_image_np)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data[0]
    if not tf_:
        output_data = output_data.transpose(1, 2, 0)

    output_data = (output_data - np.min(output_data)) / (np.max(output_data) - np.min(output_data))
    output_data = (output_data * 255.0).astype(np.uint8)
    
    return cv2.cvtColor(output_data, cv2.COLOR_BGR2RGB)