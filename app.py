# Make sure that all the following modules are already installed for use.
from flask import Flask
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np
import json

# ### Creating an instance of the flask app and an API

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

# ### Loading the trained model

HEART_ATTACK_PROGNOSIS_MODEL = joblib.load('Model/heart-attack-prognosis-model.pkl')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# ### Creating a class which is responsible for the prognosis of Lung Cancer

class HeartAttackPrognosis(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('age')
        parser.add_argument('sex')
        parser.add_argument('cp')
        parser.add_argument('trtbps')
        parser.add_argument('chol')
        parser.add_argument('fbs')
        parser.add_argument('restecg')
        parser.add_argument('thalachh')
        parser.add_argument('exng')
        parser.add_argument('oldpeak')
        parser.add_argument('slp')
        parser.add_argument('caa')
        parser.add_argument('thall')

        args = parser.parse_args()  # creates dictionary
        prognosis_input = np.fromiter(args.values(), dtype=float)  # convert input to array

        print(prognosis_input)

        out = {'Prediction': HEART_ATTACK_PROGNOSIS_MODEL.predict([prognosis_input])[0]}

        print(out)

        return json.dumps(out, cls=NpEncoder)  # returns 200 Status Code if successful with the Output


# ### Adding the predict class as a resource to the API
API.add_resource(HeartAttackPrognosis, '/prognosis_heart_attack')

# Running the Main Application
if __name__ == "__main__":
    app.run(debug=True)
    # app.run(port=5000, debug=True)