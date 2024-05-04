from flask import Flask,render_template,Response,url_for
import cv2
from ultralytics import YOLO
model=YOLO('yolov8n.pt')
video_path="moving_cars.mp4"
img=cv2.VideoCapture(video_path)
app=Flask(__name__)
def generate():
    while True:
        ret, frame = img.read()
        if not ret:
            break
        else:
            results=model(frame)
            annotated_frame=results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes  + b'\r\n')
@app.route('/')
def index():
    return render_template('object_detection.html')
@app.route("/webcam")
def webcam():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=True)
