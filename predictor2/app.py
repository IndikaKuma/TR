import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify


app = Flask(__name__)
app.config["DEBUG"] = True


#lr = joblib.load("model.pkl")

@app.route('/insurance', methods=['POST']) # path of the endpoint. Except only HTTP POST request
def predict_str():
    # the prediction input data in the message body as a JSON payload
    inout = request.get_json()
    # stringfy the JSON payload


    # age   bmi  children  sex_female  sex_male  smoker_no  smoker_yes  region_northeast  region_northwest  region_southeast  region_southwest
    # predictionArray = [
    #     inout.get('age', 0),
    #     inout.get('bmi', 0),
    #     inout.get('children', 0),
    #     inout.get('sex_female', True),
    #     inout.get('sex_male', False),
    #     inout.get('smoker_no', True),
    #     inout.get('smoker_yes', False),
    #     inout.get('region_northeast', True),
    #     inout.get('region_northwest', False),
    #     inout.get('region_southeast', False),
    #     inout.get('region_southwest', False)
    # ]
    #df = pd.DataFrame(inout, index=[0])

   # result =  lr.predict(df)
    return jsonify({'test': 'HEY!', 'result': 'true'})



# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)