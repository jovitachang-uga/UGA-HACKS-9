import pickle
import sys
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS, cross_origin

CLF_FILE_NAME = "classifier.pkl"
VEC_FILE_NAME = "vectorizer.pkl"

# classifies content as either safe or phishing
def classify(content):
    content = content.replace("%20", " ")
    print(f"content = {content}")
    clf = pickle.load(open(CLF_FILE_NAME, 'rb'))
    vectorizer = pickle.load(open(VEC_FILE_NAME, 'rb'))
    pred = clf.predict(vectorizer.transform([content]))
    return pred[0]

app = Flask(__name__)
CORS(app)

# @app.route("/?")
# def classify_output():
#     output = classify(content)
#     return render_template("template-display.html",output=output)
@app.route("/")
def home():
    return render_template('template.html')

@app.route("/get-user/<user_id>")
def get_user(user_id):
    user_id = user_id.replace("%20", " ")
    x = classify(user_id)
    user_data = {
        "type": x,
    }
    return jsonify(user_data), 200

if __name__ == "__main__":
    app.run(debug=True)