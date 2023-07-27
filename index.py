from flask import Flask
from flask_restful import Resource, Api, reqparse
import tensorflow as tf
from tensorflow import keras
from keras import layers
import requests as rq
import numpy as np

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()

image_size = (256, 256)
model = keras.models.load_model('assets/100.keras')
# keras.utils.plot_model(model, show_shapes=True)

class_names = [
    "Sarna da macieira",
    "Macieira c/ podridão negra",
    "Macieira c/ ferrugem do cedro",
    "Macieira saudável",
    "Fundo sem folha",
    "Mirtilo saudável",
    "Cerejeira c/ oídio",
    "Cerejeira saudável",
    "Milho c/ mancha cinzenta",
    "Milho c/ ferrugem comum",
    "Milho c/ mancha foliar do norte",
    "Milho saudável",
    "Videira c/ podridão negra",
    "Videira c/ sarampo negro",
    "Videira c/ mancha foliar",
    "Videira saudável",
    "Laranjeira c/ Huanglongbing",
    "Pessegueiro c/ cancro bacteriano",
    "Pessegueiro saudável",
    "Pimentão c/ cancro bacteriano",
    "Pimentão saudável",
    "Batata c/ requeima precoce",
    "Batata c/ requeima tardia",
    "Batata saudável",
    "Framboesa saudável",
    "Soja saudável",
    "Abobrinha c/ oídio",
    "Morangueiro c/ queima de folhas",
    "Tomateiro c/ mancha bacteriana",
    "Tomateiro c/ requeima precoce",
    "Tomateiro c/ requeima tardia",
    "Tomateiro c/ mofo das folhas",
    "Tomateiro c/ mancha foliar de septoria",
    "Tomateiro c/ ácaro-rajado",
    "Tomateiro com mancha-alvo",
    "Tomateiro c/ vírus da ondulação...",
    "Tomateiro c/ vírus do mosaico",
    "Tomateiro saudável",
]


class Evaluate(Resource):
    def get(self):
        response = rq.get(
            "https://expouploads231250-dev.s3.sa-east-1.amazonaws.com/public/demo.jpg")
        with open("assets/6779-new.jpg", "wb") as f:
            f.write(response.content)
        img = keras.utils.load_img(
            "assets/6779-new.jpg", target_size=image_size,
            keep_aspect_ratio=True
        )
        img.save('test.png')
        img_array = keras.utils.img_to_array(img)
        print(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        print(img_array)

        predictions = model.predict(img_array)
        print("predictions")
        print(predictions)

        k = 1
        # Get indices of top k classes
        top_classes = np.argsort(predictions[0])[::-1][:k]
        top_scores = predictions[0][top_classes]

        for i in range(k):
            class_index = top_classes[i]
            confidence = top_scores[i]
            class_name = class_names[class_index]
            print(f"Class {class_name}: Confidence = {confidence:.2%}")
        return {
            "class": class_name,
            "confidence": f"{confidence:.2%}",
        }


api.add_resource(Evaluate, '/evaluate/')
if __name__ == "__main__":
    app.run(debug=True)
