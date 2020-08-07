from wsgiref import simple_server
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
from getNumberPlateVals import detect_license_plate
from getNumberPlateVals import detect_license_plate_pytesseract
from getNumberPlateVals import detect_license_plate_easyocr
import base64
import os
# from predict_iages import DetectVehicleNumberPlate
from predict_images import DetectVehicleNumberPlate
#import easyocr
#reader = easyocr.Reader(['en'])

application = Flask(__name__)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
CORS(application)

inputFileName = "inputImage.jpg"
imagePath = "images/" + inputFileName
image_display = True
pred_stagesArgVal = 2
croppedImagepath = "images/croppedImage.jpg"


class ClientApp:
    def __init__(self):
        # modelArg = "datasets/experiment_faster_rcnn/2018_08_02/exported_model/frozen_inference_graph.pb"
        self.modelArg = "datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb"
        self.labelsArg = "datasets/records/classes.pbtxt"
        self.num_classesArg = 37
        self.min_confidenceArg = 0.5
        filepath = "autoPartsMapping/partNumbers.xlsx"
        # self.regPartDetailsObj = ReadPartDetails(filepath)
        self.numberPlateObj = DetectVehicleNumberPlate()


def decodeImageIntoBase64(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

    # return base64.b64encode(croppedImagePath)
@application.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('predict.html')

@application.route("/predict", methods=['GET', 'POST'])
def getPrediction():
    inpImage = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'images', inputFileName)
    inpImage.save(file_path)

    try:
        labelledImage = clApp.numberPlateObj.predictImages(imagePath, pred_stagesArgVal,
                                                           croppedImagepath, clApp.numberPlateObj)
        if labelledImage is not None:
            encodedCroppedImageStr = encodeImageIntoBase64(croppedImagepath)
            ig = str(encodedCroppedImageStr)
            ik = ig.replace('b\'', '')
            #numberPlateVal = detect_license_plate_pytesseract(croppedImagepath)        #Pytesseract OCR
            #numberPlateVal = detect_license_plate_easyocr(croppedImagepath, reader)    #easyOCR
            numberPlateVal = detect_license_plate(ik)
            if len(numberPlateVal) == 10:
                return numberPlateVal
            else:
                return "UnKnown"
        else:
            return "UnKnown"
    except Exception as e:
        print(e)
    return "UnKnown"


#port = int(os.getenv("PORT"))
if __name__ == '__main__':
    clApp = ClientApp()
    #host = "127.0.0.1"
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, application)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
    #application.run(host='127.0.0.1', port=port)
