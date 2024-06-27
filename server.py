import os
import os.path as pth
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
import warnings
import tempfile
from io import BytesIO

import torch
import cv2
# import ailia_tflite
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from basicsr.archs.rrdbnet_arch import RRDBNet

import config
from outpainting_gan.outpainting import outpaint_image_gan
from outpainting_sd.outpainting import outpaint_sd_overall
from super_res.sr import sr_overall
from sod.dfi import build_model, salient_crop
# from sod.back_removal import recognize_from_image
from overall_prc.overall_demo import overall_prc

os.makedirs(config.tempdir, exist_ok=True)
tempfile.tempdir = config.tempdir

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

log_file_path = pth.join(pth.dirname(pth.abspath(__file__)), 'app.log')
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

warnings.filterwarnings('ignore')


class AiModels():
    def __init__(self):
        super().__init__()
        
        # GAN
        self.gan_model = tf.lite.Interpreter(model_path=config.gan_model)
        self.gan_model.allocate_tensors()
        
        # SD
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(config.sfy_chk_model)
        self.pipe1 = StableDiffusionInpaintPipeline.from_pretrained(config.sd1_model, safety_checker=safety_checker)
        # self.pipe2 = StableDiffusionInpaintPipeline.from_pretrained(config.sd2_model, safety_checker=safety_checker)
        self.pipe1.to('cuda:3')
        # self.pipe2.to('cuda:2')
        self.sd_tagger = config.interrogator

        # SR
        self.sr1_model = tf.lite.Interpreter(model_path=config.sr1_model)
        self.sr1_model.allocate_tensors()
        
        # SOD
        self.sod_model = build_model()
        self.sod_model.load_state_dict(torch.load(config.sod1_model))
        self.sod_model.eval()
        
        # SR2_1
        self.sr2_1_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        loadnet = torch.load(config.sr2_1_model, map_location=torch.device('cpu'))
        self.sr2_1_model.load_state_dict(loadnet['params_ema' if 'params_ema' in loadnet else 'params'], strict=True)
        self.sr2_1_model.eval()
        self.sr2_1_model.to('cuda:2')

aimodels = AiModels()


def log_image_info(file_data):
    try:
        image = Image.open(BytesIO(file_data))
        app.logger.info(f'Image Info: {image.format} {image.mode} {image.size}')
    except Exception as e:
        app.logger.error(f'Error reading image data: {str(e)}')
        
@app.before_request
def log_request_info():
    request.id = uuid.uuid4()
    request.start_time = time.time()
    app.logger.info(f'Request ID: {request.id} - {request.method} {request.url}')
    app.logger.info(f'Headers: {request.headers}')
    
    if 'image' in request.files:
        image_file = request.files['image']
        file_data = image_file.read()
        log_image_info(file_data)
        image_file.seek(0)
    else:
        app.logger.info(f'Body: {request.get_data()}')

@app.after_request
def log_response_info(response):
    duration = time.time() - request.start_time
    app.logger.info(f'Request ID: {request.id} - Duration: {duration:.2f}s')
    app.logger.info(f'Response: {response.status}')
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Exception: {str(e)}', exc_info=True)
    return 'An error occurred', 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def back_remove(image, model_path):
#     interpreter = ailia_tflite.Interpreter(model_path=model_path)
#     return recognize_from_image(image, interpreter)


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify(error='No image part'), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = pth.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        method = request.form['method']
        if method not in ['original', 'gan', 'sd1', 'sd2', 'sr1', 'sr2', 'sod1', 'sod2', 'auto', 'crop', 'blur', 'retarget']:
            return jsonify(error='Invalid method'), 400

        image = cv2.imread(filepath)
        if method == 'original': processed_image = image
        if method == 'gan':  processed_image = outpaint_image_gan(image, aimodels, 128)
        if method == 'sd1':  processed_image = outpaint_sd_overall(image, aimodels)
        if method == 'sr1':  processed_image = sr_overall(image, aimodels, 128)    
        if method == 'sod1': processed_image = salient_crop(image, aimodels)
        if method == 'auto': processed_image = overall_prc(image, aimodels)
        # if method == 'sd2': processed_image = outpaint_sd_overall(image, aimodel.pipe2, aimodels.sd_tagger)
        # if method == 'sr2': processed_image = sr_overall(image,aimodels.sr1_model, 512, False)
        
        # if method == 'sod2':
        #     model_path = config.sod2_model
        #     processed_image = back_remove(image, model_path)

        _, buffer = cv2.imencode('.jpg', processed_image)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')
    
    return jsonify(error='Invalid file type'), 400


if __name__ == '__main__':    
    if not pth.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    hostIP, port = '0.0.0.0', 5006
    app.run(host=hostIP, port=port, debug=True)