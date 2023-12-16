import flask
import werkzeug
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import os
import pathlib

img_height = 180
# image width
img_width = 180

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# get dataset path
data_dir = pathlib.Path(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir)

class_names = train_ds.class_names

app = flask.Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def welcome():
    return "Hello World"\


@app.route('/predict/', methods=['GET', 'POST'])
def handle_request():


    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    img = tf.keras.utils.load_img(filename, target_size=(img_height, img_width) )
    # image to array
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch




    loaded_model = models.load_model('model.keras')
    predictions = loaded_model.predict(img_array)
    # get score
    score = tf.nn.softmax(predictions[0])

    print(
        "La imagen pertenece a: {} con un {:.2f} % de Confianza"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    print(class_names)
    print(np.argmax(score))

    return str("La imagen pertenece a: {} con un {:.2f} % de confianza"
        .format(class_names[np.argmax(score)], 100 * np.max(score)))


app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000), debug=True)