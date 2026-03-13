from flask import Flask,render_template,request,jsonify
import cv2
import numpy as np
import pickle

app=Flask(__name__)

model=pickle.load(open("rf_parkinsons_model.pkl","rb"))

def extract_features(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur=cv2.GaussianBlur(gray,(5,5),0)

    edges=cv2.Canny(blur,50,150)

    feature=np.count_nonzero(edges)

    return feature


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():

    file=request.files["file"]

    file_bytes=np.frombuffer(file.read(),np.uint8)

    img=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    feature=extract_features(img)

    feature=np.array([[feature]])

    prediction=model.predict(feature)

    if prediction[0]==0:
        result="Healthy"
    else:
        result="Parkinson Risk"

    return jsonify({"result":result})


if __name__=="__main__":
    app.run(debug=True)