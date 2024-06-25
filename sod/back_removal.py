
# import cv2
# import numpy as np
# import ailia_tflite

# from util.model_utils import check_and_download_models, format_input_tensor
# from util.u2net_utils import imread, load_image, norm, save_result, transform
# from util.utils import get_base_parser, get_savepath, update_parser, delegate_obj


# def recognize_from_image(image, interpreter):

#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     image_path = image
#     input_data, h, w = load_image(image_path, scaled_size=(320, 320), rgb_mode=True)

#     inputs = format_input_tensor(input_data, input_details, 0)
#     interpreter.set_tensor(input_details[0]['index'], inputs)
#     interpreter.invoke()

#     details = output_details[0]
#     dtype = details['dtype']
#     if dtype == np.uint8 or dtype == np.int8:
#         quant_params = details['quantization_parameters']
#         int_tensor = interpreter.get_tensor(details['index'])
#         real_tensor = int_tensor - quant_params['zero_points']
#         real_tensor = real_tensor.astype(np.float32) * quant_params['scales']
#     else:
#         real_tensor = interpreter.get_tensor(details['index'])

#     return real_tensor
