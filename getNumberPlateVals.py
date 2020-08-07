import base64
import requests
import json
import re
#import pytesseract         #Uncomment for Pytesserect
import shutil
import os
import random
from PIL import Image


url = 'https://vision.googleapis.com/v1/images:annotate?key="YOUR GOOGLE VISION API KEY"'




def detect_license_plate(encodedImage):
    # img_base64 = base64.b64encode(imagePath)
    headers = {'content-type': 'application/json'}

    data = """{
      "requests": [
        {
          "image": {
                   "content": '""" + encodedImage[:-1] + """'

                    },

          "features": [
            {
              "type": "TEXT_DETECTION"
            }
          ]
        }
      ]
    }"""
    r = requests.post(url, headers=headers, data=data)
    result = json.loads(r.text)
    print(result)
    try:
        result = result['responses'][0]['textAnnotations'][0]['description']
    except Exception as e:
        return r
    result = result.replace('\n', '').replace(' ', '')
    result =re.sub('\W+','', result)
    mystates = ['AP','AR','AS','BR','CG','GA','GJ','HR' ,'HP' ,'JK','JH','KA','KL','MP','MH','MN','ML','MZ','NL' ,'OD','PB' ,'RJ','SK','TN','TS','TR','UA','UK','UP','WB','AN','CH','DN','DD','DL' ,'LD','PY']
    
    if(len(result) > 0):
        #res=re.findall("\s*[AP,AR,AS,BR,CG,GA,GJ,HR,HP,JK,JH,KA,KL,MP,MH,MN,ML,MZ,NL,OD,PB,RJ,SK,TN,TS,TR,UA,UK,UP,WB,AN,CH,DN,DD,DL,LD,PY]{2}\s*[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?",cleanString)    
        for word in mystates:
            if(word in result):
                res = re.findall(word + "[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?", result)
                if(len(res) >0):
                    return(res[0])
    return result


def detect_license_plate_pytesseract(imagePath):
    try:
        text = pytesseract.image_to_string(Image.open(imagePath))
        text = re.sub('\W+', '', text)
        mystates = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN',
                    'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UA', 'UK', 'UP', 'WB', 'AN', 'CH',
                    'DN', 'DD', 'DL', 'LD', 'PY']

        if (len(text) > 0):
            # res=re.findall("\s*[AP,AR,AS,BR,CG,GA,GJ,HR,HP,JK,JH,KA,KL,MP,MH,MN,ML,MZ,NL,OD,PB,RJ,SK,TN,TS,TR,UA,UK,UP,WB,AN,CH,DN,DD,DL,LD,PY]{2}\s*[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?",cleanString)
            for word in mystates:
                if (word in text):
                    res = re.findall(word + "[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?", text)
                    if (len(res) > 0):
                        return (res[0])
        return text
    except Exception as e:
        raise e


def detect_license_plate_easyocr(imagePath, reader):
    text = reader.readtext(imagePath)
    print(text)
    text = text[0][1]
    text = re.sub('\W+', '', text)
    mystates = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN',
                'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UA', 'UK', 'UP', 'WB', 'AN', 'CH',
                'DN', 'DD', 'DL', 'LD', 'PY']

    if (len(text) > 0):
        # res=re.findall("\s*[AP,AR,AS,BR,CG,GA,GJ,HR,HP,JK,JH,KA,KL,MP,MH,MN,ML,MZ,NL,OD,PB,RJ,SK,TN,TS,TR,UA,UK,UP,WB,AN,CH,DN,DD,DL,LD,PY]{2}\s*[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?",cleanString)
        for word in mystates:
            if (word in text):
                res = re.findall(word + "[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?", text)
                if (len(res) > 0):
                    return (res[0])
    return text





