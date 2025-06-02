from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from sync_and_generate_video import generate_final_montage
from detect_kills import detect_kills
from extract_clips import extract_kill_clips
import uuid
import shutil
import time
from datetime import datetime
import logging
import cv2
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
CLIPS_FOLDER = 'kill_clips'
OUTPUT_FOLDER = 'output'
MODEL_PATH = "best.pt"  # Path to your YOLO model

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Store processing status
processing_status = {}

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    model = None

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi'}

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav'}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """Upload raw gameplay video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_video_file(file.filename):
            # Clear existing clips
            shutil.rmtree(CLIPS_FOLDER)
            os.makedirs(CLIPS_FOLDER)

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return jsonify({
                'message': 'Video uploaded successfully',
                'filename': filename,
                'size': os.path.getsize(file_path)
            })

        return jsonify({'error': 'Invalid file type. Allowed types: MP4, MOV, AVI'}), 400

    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/audio', methods=['POST'])
def upload_audio():
    """Upload background music"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_audio_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return jsonify({
                'message': 'Audio uploaded successfully',
                'filename': filename,
                'size': os.path.getsize(file_path)
            })

        return jsonify({'error': 'Invalid file type. Allowed types: MP3, WAV'}), 400

    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_video():
    """Process the video with kill detection and montage generation"""
    try:
        data = request.get_json()
        
        if not data or 'video_filename' not in data or 'audio_filename' not in data:
            return jsonify({'error': 'Video and audio filenames are required'}), 400

        video_filename = secure_filename(data['video_filename'])
        audio_filename = secure_filename(data['audio_filename'])
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404

        # Generate unique output filename and job ID
        job_id = str(uuid.uuid4())
        output_filename = f'montage_{job_id}.mp4'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Initialize processing status
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'start_time': time.time(),
            'output_filename': output_filename,
            'stage': 'detecting_kills'
        }

        try:
            # Start processing in a separate thread
            def process_task():
                try:
                    # 1. Detect kills
                    processing_status[job_id]['stage'] = 'detecting_kills'
                    processing_status[job_id]['progress'] = 10
                    
                    # Create timestamps file path
                    timestamps_file = os.path.join(app.config['UPLOAD_FOLDER'], f'kill_timestamps_{job_id}.txt')
                    
                    # Detect kills and save timestamps
                    kill_times = detect_kills(video_path, timestamps_file)
                    logger.info(f"Detected {len(kill_times)} kills")

                    # 2. Extract kill clips
                    processing_status[job_id]['stage'] = 'extracting_clips'
                    processing_status[job_id]['progress'] = 30
                    kill_clips = extract_kill_clips(video_path, timestamps_file)
                    logger.info(f"Extracted {len(kill_clips)} kill clips")

                    # 3. Generate montage
                    processing_status[job_id]['stage'] = 'generating_montage'
                    processing_status[job_id]['progress'] = 50
                    generate_final_montage(
                        clips_folder=CLIPS_FOLDER,
                        music_path=audio_path,
                        output_path=output_path
                    )
                    logger.info("Montage generation completed")

                    processing_status[job_id]['status'] = 'completed'
                    processing_status[job_id]['progress'] = 100
                    processing_status[job_id]['end_time'] = time.time()
                    processing_status[job_id]['kill_count'] = len(kill_times)
                    processing_status[job_id]['clip_count'] = len(kill_clips)

                    # Clean up timestamps file
                    if os.path.exists(timestamps_file):
                        os.remove(timestamps_file)

                except Exception as e:
                    processing_status[job_id]['status'] = 'failed'
                    processing_status[job_id]['error'] = str(e)
                    logger.error(f"Processing error: {str(e)}")

            import threading
            thread = threading.Thread(target=process_task)
            thread.start()
            
            return jsonify({
                'message': 'Processing started',
                'job_id': job_id
            })

        except Exception as e:
            processing_status[job_id]['status'] = 'failed'
            processing_status[job_id]['error'] = str(e)
            raise e

    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the status of a processing job"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download the generated video"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Clean up temporary files"""
    try:
        # Clean up uploads
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        
        # Clean up clips
        shutil.rmtree(CLIPS_FOLDER)
        os.makedirs(CLIPS_FOLDER)
        
        # Clean up output
        for file in os.listdir(OUTPUT_FOLDER):
            os.remove(os.path.join(OUTPUT_FOLDER, file))
        
        # Clear processing status
        processing_status.clear()
        
        return jsonify({'message': 'Cleanup completed successfully'})
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 