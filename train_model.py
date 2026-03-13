import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

X=[]
y=[]

base_path="archive/spiral/training"

for label,folder in enumerate(["healthy","parkinson"]):

    path=os.path.join(base_path,folder)

    for img_name in os.listdir(path):

        img_path=os.path.join(path,img_name)

        img=cv2.imread(img_path)

        if img is None:
            continue

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        blur=cv2.GaussianBlur(gray,(5,5),0)

        edges=cv2.Canny(blur,50,150)

        feature=np.count_nonzero(edges)

        X.append(feature)
        y.append(label)

X=np.array(X).reshape(-1,1)
y=np.array(y)

model=RandomForestClassifier(n_estimators=100)

model.fit(X,y)

pickle.dump(model,open("rf_parkinsons_model.pkl","wb"))

print("Model trained successfully")