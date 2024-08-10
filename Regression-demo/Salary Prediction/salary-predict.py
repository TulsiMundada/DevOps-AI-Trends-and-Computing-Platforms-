from flask import Flask , request , jsonify
import joblib 
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("final_salary_model.pkl")
col_names = joblib.load("salary_column_names.pkl")

@app.route('/')
def hello() -> str:
    return "Hello, World!"

@app.route('/predict' , methods = ['POST'])
def predict():
    user_data = request.json
    df = pd.DataFrame(user_data)
    df = df.reindex(columns=col_names)
    prediction = list(model.predict(df))
    print(type(prediction))
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')