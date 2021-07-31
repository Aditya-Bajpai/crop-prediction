from flask import Flask , render_template , request
import numpy as np
import  pickle
app = Flask(__name__)

model = pickle.load(open('model.pkl' , 'rb'))


@app.route("/")
def main():
    return render_template("main.html")

@app.route("/predict" , methods=["GET", "POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template("main.html" , prediction_text = f"You should grow {output} on your field")
    

if __name__ == "__main__":
    debug = True
