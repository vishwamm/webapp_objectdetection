from flask import Flask,render_template,Response,url_for
import cv2
from ultralytics import YOLO
model=YOLO('yolov8n.pt')
img=cv2.imread("cr7.jpg")
app=Flask(__name__)
def generate():
    results=model(img)
    annotated_frame=results[0].plot()
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_bytes = buffer.tobytes()
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes  + b'\r\n')
@app.route('/')
def index():
    return render_template('object_detection.html',images=img)
@app.route("/webcam")
def webcam():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=True)
