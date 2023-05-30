from flask import Flask
from flask_restful import Resource, Api, reqparse
import tensorflow as tf
from tensorflow import keras
from keras import layers
import requests as rq

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
STUDENTS = {
    '1': {'name': 'Mark', 'age': 23, 'spec': 'math'},
    '2': {'name': 'Jane', 'age': 20, 'spec': 'biology'},
    '3': {'name': 'Peter', 'age': 21, 'spec': 'history'},
    '4': {'name': 'Kate', 'age': 22, 'spec': 'science'},
}

image_size = (180, 180)
model = keras.models.load_model('assets/test_save_2')
# keras.utils.plot_model(model, show_shapes=True)


class StudentsList(Resource):
    def post(self):
        parser.add_argument("name")
        parser.add_argument("age")
        parser.add_argument("spec")
        args = parser.parse_args()
        student_id = int(max(STUDENTS.keys())) + 1
        student_id = '%i' % student_id
        STUDENTS[student_id] = {
            "name": args["name"],
            "age": args["age"],
            "spec": args["spec"],
        }
        return STUDENTS[student_id], 201

    def get(self):
        response = rq.get(
            "https://expouploads231250-dev.s3.sa-east-1.amazonaws.com/public/demo.jpg")
        with open("assets/6779-new.jpg", "wb") as f:
            f.write(response.content)
        img = keras.utils.load_img(
            "assets/6779-new.jpg", target_size=image_size
        )
        img_array = keras.utils.img_to_array(img)
        print("1 img array")
        print(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        print("2 img array")
        print(img_array)

        predictions = model.predict(img_array)
        print("predictions")
        print(predictions)
        score = float(predictions[0])
        print(
            f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
        return {
            "cat": 100 * (1 - score),
            "dog": 100 * score
        }


api.add_resource(StudentsList, '/students/')
if __name__ == "__main__":
    app.run(debug=True)
