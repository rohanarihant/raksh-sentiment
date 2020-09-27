import os
from flask import Flask, request, jsonify
from lime_explainer import explainer, tokenizer, METHODS

app = Flask(__name__)
SECRET_KEY = os.urandom(24)

@app.route('/result', methods=['POST'])
def index():
    exp = ""
    if request.method == 'POST':
        text = tokenizer(request.form['entry'])
        method = "textblob"
        n_samples = 2000
        if any(not v for v in [text, n_samples]):
            return jsonify(success=False, message= "Please do not leave text fields blank.")
        if method != "base":
            exp = explainer(method,
                            path_to_file=METHODS[method]['file'],
                            text=text,
                            num_samples=int(n_samples))
            exp = exp.available_labels()[0]+1
        return jsonify(success=True, sentiment=int(exp), message="Success")

if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)
