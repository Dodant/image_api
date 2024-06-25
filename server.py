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
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

import config
from outpainting_gan.outpainting import *
from outpainting_sd.outpainting import outpaint_sd_overall
from super_res.sr import sr_overall
from sod.dfi import build_model
# from sod.back_removal import recognize_from_image

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


def log_image_info(file_data):
    try:
        image = Image.open(BytesIO(file_data))
        app.logger.info(f'Image format: {image.format}')
        app.logger.info(f'Image size: {image.size}')
        app.logger.info(f'Image mode: {image.mode}')
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
    app.logger.info(f'Request ID: {request.id} - Duration: {duration:.3f}s')
    app.logger.info(f'Response: {response.status}')
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Exception: {str(e)}', exc_info=True)
    return 'An error occurred', 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def blur_image(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def outpaint_image_gan(image, model_path, input_size=128):
    interpreter = load_tflite_model(model_path)
    masked_image = resize_masking(image, (input_size, input_size))
    output_image = predict_image(interpreter, masked_image)
    output_image = postprocess_image(output_image)
    
    _, processed_image = outpaint(output_image, image)
    return (processed_image * 255).astype('uint8')

def salient_crop(image_original, model_path):
    model = build_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    in_ = np.array(image_original, dtype=np.float32) 
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    image, _ = in_.transpose((2,0,1)), tuple(in_.shape[:2])
    image = torch.Tensor(image).unsqueeze(0)
    
    with torch.no_grad():
        preds = model(image, mode=3)
        pred_sal = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
        pred_sal = 255 * pred_sal
        pred_sal = np.where(pred_sal > 126, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(pred_sal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        for _, (x, y, w, h) in enumerate(bounding_boxes):
            return image_original[y:y+h, x:x+w] 
        
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
        if method not in ['gan', 'sd1', 'sd2', 'sr1', 'sr2', 'sod1', 'sod2', 'crop', 'blur', 'retarget']:
            return jsonify(error='Invalid method'), 400

        image = cv2.imread(filepath)
        if method == 'gan':
            model_path = config.gan_model
            processed_image = outpaint_image_gan(image, model_path, 128)

        if method == 'sd1':
            model_path = config.sd1_model
            processed_image = outpaint_sd_overall(image, model_path)
    
        if method == 'sd2':
            model_path = config.sd2_model
            processed_image = outpaint_sd_overall(image, model_path)
    
        if method == 'sr1':
            model_path = config.sr1_model
            processed_image = sr_overall(image, model_path, 128)
            
        if method == 'sr2':
            model_path = config.sr2_model
            processed_image = sr_overall(image, model_path, 512, False)
        
        if method == 'sod1':
            model_path = config.sod1_model
            processed_image = salient_crop(image, model_path)
            
        # if method == 'sod2':
        #     model_path = config.sod2_model
        #     processed_image = back_remove(image, model_path)
        
        if method == 'crop':
            x, y = int(request.form.get('x', 0)), int(request.form.get('y', 0))
            w, h = int(request.form.get('w', 100)), int(request.form.get('h', 100))
            processed_image = crop_image(image, x, y, w, h)
            
        if method == 'blur':
            ksize = int(request.form.get('ksize', 5))
            if ksize % 2 == 0: ksize += 1
            processed_image = blur_image(image, ksize)  
            
        if method == 'retarget':
            pass

        if method == 'roi_crop':
            pass

        _, buffer = cv2.imencode('.jpg', processed_image)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')
    
    return jsonify(error='Invalid file type'), 400


if __name__ == '__main__':    
    if not pth.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    hostIP, port = '0.0.0.0', 5006
    app.run(host=hostIP, port=port, debug=True)

