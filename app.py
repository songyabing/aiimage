import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import configparser
import threading
import queue
import multiprocessing  # 导入 multiprocessing
import logging

# 读取配置文件
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# static 文件夹路径
STATIC_DIR = os.path.join(BASE_DIR, 'static')
# 上传文件夹路径
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')

MODEL_PATH = config['DEFAULT'].get('MODEL_PATH', 'Salesforce/blip-image-captioning-base')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 限制上传文件大小为 4MB

# 全局变量初始化
processor = None
model = None
results_queue = queue.Queue()

app.logger.setLevel(logging.INFO)

# 如果需要更详细的日志，可以添加文件处理器
if not app.debug:
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

def init_model():
    """初始化模型和处理器"""
    global processor, model
    if processor is None or model is None:
        processor = BlipProcessor.from_pretrained(MODEL_PATH)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_caption(file_path, result_queue):
    """处理单个文件并将结果放入队列"""
    try:
        if processor is None or model is None:
            init_model()
        
        with Image.open(file_path).convert("RGB") as image:
            inputs = processor(image, return_tensors="pt")
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            result_queue.put(caption)
    except Exception as e:
        result_queue.put(f"处理图片时出错: {str(e)}")
    finally:
        # 确保释放资源
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs

@app.route('/')
def index():
    return render_template('index.html')

def ensure_upload_folder():
    """确保上传文件夹存在并可访问"""
    try:
        # 先确保 static 文件夹存在
        os.makedirs(STATIC_DIR, exist_ok=True)
        # 再确保 uploads 文件夹存在
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        app.logger.info(f"上传文件夹路径: {UPLOAD_FOLDER}")
    except Exception as e:
        app.logger.error(f"创建文件夹失败: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # 确保上传文件夹存在
        ensure_upload_folder()
        
        if 'files' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['files']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400

        # 生成安全的文件名
        filename = secure_filename(file.filename)
        # 确保使用正确的路径拼接
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 添加调试日志
        app.logger.info(f"尝试保存文件到: {filepath}")
        
        # 确保文件夹存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存文件
        file.save(filepath)
        
        # 验证文件是否成功保存
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件保存失败: {filepath}")
            
        app.logger.info(f"文件成功保存到: {filepath}")

        # 处理图片生成描述
        result_queue = queue.Queue()
        thread = threading.Thread(target=generate_caption, args=(filepath, result_queue))
        thread.start()
        thread.join(timeout=30)

        if thread.is_alive():
            raise TimeoutError("处理超时")

        caption = result_queue.get_nowait()
        if caption.startswith("处理图片时出错"):
            raise Exception(caption)

        return jsonify({
            'captions': [{
                'caption': caption
            }]
        })

    except FileNotFoundError as e:
        app.logger.error(f"文件未找到: {str(e)}")
        return jsonify({'error': f'文件保存失败: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"处理上传时出错: {str(e)}")
        return jsonify({'error': f'处理失败: {str(e)}'}), 500
    finally:
        # 清理临时文件
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
                app.logger.info(f"已清理临时文件: {filepath}")
        except Exception as e:
            app.logger.error(f"清理文件失败: {str(e)}")

@app.route('/inspiration')
def inspiration():
    return render_template('inspiration.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

if __name__ == '__main__':
    print(f"项目根目录: {BASE_DIR}")
    print(f"Static文件夹: {STATIC_DIR}")
    print(f"上传文件夹: {UPLOAD_FOLDER}")
    
    ensure_upload_folder()
    init_model()
    app.run(host='0.0.0.0', port=8080, debug=True)
