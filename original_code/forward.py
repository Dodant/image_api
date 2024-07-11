# Basile Van Hoorick, Jan 2020
'''
Hallucinates beyond all four edges of an image, increasing both dimensions by 50%.
The outpainting process interally converts 128x128 to 192x192, after which the generated output is upscaled.
Then, the original input is blended onto the result for optimal fidelity.
Example usage:
python forward.py input.jpg output.jpg
'''

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sys
    from outpainting import *

    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    src_file = sys.argv[1]
    # dst_file = sys.argv[2]
    # model_selection = sys.argv[3]
    # gen_model = load_model('generator_final.pt')

    for model_selection in ['art','nat','rec']:
        st = time.time()
        # try: model_selection = sys.argv[2]
        # except: model_selection = 'nat'
        gen_model = load_model(f'models/G_{model_selection}.pt')
            
        print(f'Source file: {src_file} ...')
        input_img = plt.imread(src_file)[:, :, :3]
        output_img, blended_img = perform_outpaint(gen_model, input_img)
        output_save_name = f'{src_file.split(".")[:-1][0]}_infer_output_{model_selection}.jpg'
        blended_save_name = f'{src_file.split(".")[:-1][0]}_infer_blended_{model_selection}.jpg'
        plt.imsave(output_save_name, output_img)
        plt.imsave(blended_save_name, blended_img)
        print(f'Output file: {output_save_name} written')
        print(f'Blended file: {blended_save_name} written')
        print(f'Total Time: {time.time() - st:.2f}')