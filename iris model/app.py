from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('iris_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract form inputs and convert to float
            sl = float(request.form['sepal_length'])
            sw = float(request.form['sepal_width'])
            pl = float(request.form['petal_length'])
            pw = float(request.form['petal_width'])
            features = [[sl, sw, pl, pw]]
            
            prediction = int(model.predict(features)[0])
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
