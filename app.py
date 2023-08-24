from flask import Flask, request
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def hello():
    return "Hello World"

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files['image']
    image_name = request.form['imageName']

    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    height, width = img.shape
    
    image_name = image_name.split('.')[0]
    try:
        with open("annotations/" + image_name + '.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except:
        return "Not a Valid Image"

    text = ''
    prev_y_center = 0
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces

        # If the line is empty, it indicates a space
        if not line:
            text += ' '
            continue

        line_data = line.split()
        char, x_center, y_center, w, h = line_data[0], float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4])

        # Denormalize the coordinates
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height

        # If y_center has changed significantly, it means we moved to the next line
        if y_center - prev_y_center > h:
            text += '\n'
        prev_y_center = y_center

        # Add character to the text
        text += char

    return text

if __name__ == '__main__':
    app.run(port=8080, debug=True, host="0.0.0.0", threaded=True, use_reloader=True, passthrough_errors=True)
