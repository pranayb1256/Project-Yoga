from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
from utils import predict_video  # Assuming your utils.py is imported
import joblib

app = Flask(__name__)

model = joblib.load('yoga_pose_model.pkl')

# Test phase : build test dataset then evaluate
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the file temporarily
    video_path = "temp_video.mp4"
    file.save(video_path)
    
    # Run analysis on the video
    result = predict_video(model, video_path, show=False)  # Adjust based on your function
    
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, port=8000
