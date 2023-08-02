#Install Libraries

from flask import Flask, request, jsonify
from tensorflow import keras
from keras.utils import pad_sequences
from string import printable
import traceback
import pandas as pd
import numpy as np
import sys

application = Flask(__name__)

@application.route('/prediction', methods=['POST'])

def predict():
    if lstm and convlstm:
        try:
            query = request.json["review"][0]
            url_int_tokens = [[printable.index(x) + 1 for x in query if x in printable]]
            print("URL INT TOKENS is ", url_int_tokens, " till here")
            query = pad_sequences(url_int_tokens, maxlen=75)
            # query = query.reindex(columns=lstm.columns, fill_value=0)   
            print("Query is: ", query, " till here.")
            mal_predict = lstm.predict(query)
            phish_predict = convlstm.predict(query)
            mal_predict = round(mal_predict[0][0])
            phish_predict = round(phish_predict[0][0])
            print("Malicious prediction", mal_predict, "and Phishing prediction", phish_predict)
            if (mal_predict > 0.5):
                if (phish_predict > 0.5):
                    res = "The website is not malicious and it is not phishing."
                else:
                    res = "The website is not malicious but it is phishing."
            else:
                if (phish_predict > 0.5):
                    res = "The website is malicious in nature but it is not phishing."
                else:
                    res = "The website is malicious in nature and it is also phishing."

            return jsonify({'prediction': res})
        
        except:
            return jsonify({'trace': traceback.format_exc()})
    
    else:
        return ('Model is not good')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    
    except:
        port = 12345
    
    lstm = keras.models.load_model("lstm.h5")
    convlstm = keras.models.load_model("convlstm.h5")
    application.run(port=port, debug=True)