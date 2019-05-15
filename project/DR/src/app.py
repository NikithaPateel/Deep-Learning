import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import time
import matplotlib.pyplot as plt

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT,'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = '/'.join([target,filename])
        print(destination)
        file.save(destination)

        model = load_model('retinopathy_predict.h5')
        img = image.load_img(destination, target_size=model.input_shape[1:])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='keras')
        start_time1 = int(round(time.time() * 1000))
        preds = model.predict(x)
        print(preds)
        output = np.argmax(preds)
        x = preds[0]
        num_bins = 5
        y = ['NO DR',' Mild DR', 'Moderate DR', 'Severe DR','Proliferate DR']
        # n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        # plt.show()

        plot =plt.bar(y, x)
        plt.ylabel('Probability')
        plt.xlabel('classes')
        plt.title('Diabetic Retinopathy')
        plt.xticks(fontsize=5, rotation=30)
        plt.yticks(fontsize=5)

        out = os.path.join(APP_ROOT, 'static/')
        plt.savefig(out+'css/pred.jpg')
        plt.show()

    return render_template('complete.html')

if __name__ == "__main__":
    app.run(port=4555 ,debug=True)












