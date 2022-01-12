from re import I
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import splitfolders

input_dir = "simpsons_dataset"

output_dir = "/cbir-flask-char/set/data"

splitfolders.ratio(input_dir, output=output_dir,seed=1337, ratio=(.9, 0.1))

train_dir = "/cbir-flask-char/set/data/train"
val_dir = "/cbir-flask-char/set/data/val"
test_dir = "/cbir-flask-char/set/data/test"

labels = os.listdir(train_dir, model_dir='/tmp/set/data/train')
app = Flask(__name__)



MODEL_PATH = 'models/model.h5'
model = load_model(MODEL_PATH)

model.make_predict_function()




# routes
@app.route("/", methods=['GET'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['POST'])
def predict():
	img = request.files['my_image']
	img_path = "uploads/" + img.filename		
	img.save(img_path)
	
	i = image.load_img(img_path, target_size=(64, 64 ))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 64,64,3)
	p = model.predict(i)
	p = labels[np.argmax(p)]
	
	

		
	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)