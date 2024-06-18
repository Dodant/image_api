import tempfile
import os, time, uuid
from io import BytesIO
from PIL import Image
import warnings, logging
from logging.handlers import RotatingFileHandler

import cv2
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

from outpainting_gan.outpainting import *
from outpainting_sd.outpainting import outpaint_sd
from super_res.sr import sr_overall

tempdir = '/home/compu/Downloads/tmp2/temp'
os.makedirs(tempdir, exist_ok=True)
tempfile.tempdir = tempdir

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log')
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
        app.logger.info(f"Image format: {image.format}")
        app.logger.info(f"Image size: {image.size}")
        app.logger.info(f"Image mode: {image.mode}")
    except Exception as e:
        app.logger.error(f"Error reading image data: {str(e)}")
        
@app.before_request
def log_request_info():
    request.id = uuid.uuid4()
    request.start_time = time.time()
    app.logger.info(f"Request ID: {request.id} - {request.method} {request.url}")
    app.logger.info(f"Headers: {request.headers}")
    
    if 'image' in request.files:
        image_file = request.files['image']
        file_data = image_file.read()
        log_image_info(file_data)
        image_file.seek(0)
    else:
        app.logger.info(f"Body: {request.get_data()}")

@app.after_request
def log_response_info(response):
    duration = time.time() - request.start_time
    app.logger.info(f"Request ID: {request.id} - Duration: {duration:.3f}s")
    app.logger.info(f"Response: {response.status}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Exception: {str(e)}", exc_info=True)
    return "An error occurred", 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def blur_image(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def outpaint_image_gan(image, input_size=128, model_path = 'models/G_rec_fp32.tflite'):
    interpreter = load_tflite_model(model_path)
    masked_image = resize_masking(image, (input_size, input_size))
    output_image = predict_image(interpreter, masked_image)
    output_image = postprocess_image(output_image)
    
    _, processed_image = outpaint(output_image, image)
    return (processed_image * 255).astype('uint8')

@app.route('/process_image', methods=['POST'])
def process_image():
    
    if 'image' not in request.files:
        return jsonify(error='No image part'), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        method = request.form['method']
        if method not in ['outpaint_gan',  'outpaint_sd1', 'outpaint_sd2', 'super_resolution1', 'super_resolution2',
                          'gan', 'sd1', 'sd2', 'sr1', 'sr2', 'crop', 'blur', 'retarget']:
            return jsonify(error='Invalid method'), 400

        image = cv2.imread(filepath)
        if method in ['outpaint_gan', 'gan']:
            processed_image = outpaint_image_gan(image)

        elif method in ['outpaint_sd1', 'sd1']:
            processed_image = outpaint_sd(image)
        
        elif method in ['outpaint_sd2', 'sd2']:
            processed_image = outpaint_sd(image, False)
    
        elif method in ['super_resolution1', 'sr1']:
            processed_image = sr_overall(image, 'models/esrgan.tflite', 128)
            
        elif method in ['super_resolution2', 'sr2']:
            processed_image = sr_overall(image, 'models/RealESRGAN.tflite', 512, False)
        
        elif method == 'crop':
            x, y = int(request.form.get('x', 0)), int(request.form.get('y', 0))
            w, h = int(request.form.get('w', 100)), int(request.form.get('h', 100))
            processed_image = crop_image(image, x, y, w, h)
            
        elif method == 'blur':
            ksize = int(request.form.get('ksize', 5))
            if ksize % 2 == 0: ksize += 1
            processed_image = blur_image(image, ksize)  
            
        elif method == 'retarget':
            pass

        elif method == "roi_crop":
            pass
        
        _, buffer = cv2.imencode('.jpg', processed_image)
        
        return send_file(BytesIO(buffer), mimetype='image/jpeg')
    
    return jsonify(error='Invalid file type'), 400


if __name__ == '__main__':    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    hostIP, port = '0.0.0.0', 5005
    app.run(host=hostIP, port=port, debug=True)

