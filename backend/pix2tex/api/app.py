
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from http import HTTPStatus
from PIL import Image
from io import BytesIO
import sys
# todo: get import dir
pd = os.path.dirname(os.getcwd())
pd = os.path.dirname(pd)

sys.path.append(pd)
print("sys.path",sys.path)
from pix2tex.cli import LatexOCR,LatexOCR_my,LatexOCR_my_non_resize
model = None
app = Flask(__name__)
CORS(app)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image




@app.get('/')
def root():
    '''Health check.'''
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {},
    }
    print('===root')
    return response



# Route to check connection


@app.route("/")
def hello():
    global model
    print("model")
    if model is None:
        print("LatexOCR_my_non_resize")
        model = LatexOCR_my_non_resize()
    return "Everything looks good"

# Endpoint to get image from client-side


@app.route('/predict/', methods=['POST'])
def getImage():
    if request.method == "POST":
        file = request.files['imageFile']
        # model_name = request.files['model_version']
        image = Image.open(file)
        # print("image",image)
        global model
        model = LatexOCR_my_non_resize()
        # print("======LatexOCR_my_non_resize()")
            # print("LatexOCR()")
            # if model_name == 'new_model':
            #     print("LatexOCR_my_non_resize()")
            #     model = LatexOCR_my_non_resize()
            # else:
            #     model = LatexOCR()
            #     print("LatexOCR")

        result = model(image)
        # print("=== result_0", result)
        result = result.replace("\\noalign{\smallskip}", "")
        result = result.replace("\mbox", "\ \\text")
        i = result.find("\ \\text{")
        if i != -1:
            for j in range(i,len(result)):
                if result[j] =='}':
                    result = result[:j] + '\ ' + result[j:]
                    break
        # print("=== result", result)
        return jsonify({'prediction': result})

@app.route('/predict_old/', methods=['POST'])
def getImage_old():
    if request.method == "POST":
        file = request.files['imageFile']
        # model_name = request.files['model_version']
        image = Image.open(file)
        # print("image",image)
        global model
        model = LatexOCR()
        # print("======LatexOCR()")
            # print("LatexOCR()")
            # if model_name == 'new_model':
            #     print("LatexOCR_my_non_resize()")
            #     model = LatexOCR_my_non_resize()
            # else:
            #     model = LatexOCR()
            #     print("LatexOCR")

        result = model(image)
        # print("=== result_0", result)
        result = result.replace("\\noalign{\smallskip}", "")
        result = result.replace("\mbox", "\ \\text")
        i = result.find("\ \\text{")
        if i != -1:
            for j in range(i,len(result)):
                if result[j] =='}':
                    result = result[:j] + '\ ' + result[j:]
                    break
        # print("=== result", result)
        return jsonify({'prediction': result})


if __name__ == "__main__":
    app.run(port=8001)
