import sys, os
import time

import matplotlib.pyplot as plt

from outpainting_gan.outpainting import *


if __name__ == '__main__':

    image_path = sys.argv[1]
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    for i in ['art','nat','rec']:
        for j in ['fp32','fp16','int8']:
            st = time.time()
            
            model_path = f'models/G_{i}_{j}.tflite'
            input_img = plt.imread(image_path)[:, :, :3]
            output_image_path = f'{image_name}/{i}_{j}_blend.jpg'
            interpreter = load_tflite_model(model_path)
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            input_shape = input_details[0]['shape']
            input_quant_params = input_details[0]['quantization']
            output_quant_params = output_details[0]['quantization']
            masked_img = resize_masking(input_img, (input_size, input_size))
            
            if j == 'int8':
                quantization_params = {
                    'scale': input_quant_params[0],
                    'zero_point': input_quant_params[1]
                }    
                output_quantization_params = {
                    'scale': output_quant_params[0],
                    'zero_point': output_quant_params[1]
                }       
                input_image = preprocess_image_for_int8(masked_img, quantization_params, True)
                output_img = predict_image(interpreter, input_image)
                output_img = postprocess_image(output_img, output_quantization_params, True)
            else:
                output_img = predict_image(interpreter, masked_img)
                output_img = postprocess_image(output_img)
            
            output_img, blended_img = outpaint(output_img, input_img)
            blended_scaled = (blended_img * 255).astype('uint8')
            
            blended_img = Image.fromarray(blended_scaled)
            blended_img_path = f'{image_name}/{i}_{j}_blend.jpg'
            blended_img.save(blended_img_path)
            print(f'Blended file: {image_name}/{i}_{j}_output.jpg written')

            output_img = Image.fromarray(output_img)
            output_img_path = f'{image_name}/{i}_{j}_output.jpg'
            output_img.save(output_img_path)
            print(f'Output file: {image_name}/{i}_{j}_blend.jpg written')
            
            print(f'Total Time: {time.time() - st:.2f}')