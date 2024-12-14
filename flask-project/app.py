from flask import Flask, render_template, request
from optimize import optimize

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/about')
def subpage2():
    return render_template('about.html')

@app.route('/optimize', methods=['GET', 'POST'])
def run_optimization():
    if request.method == 'GET':
        return render_template('optimize_form.html')
    elif request.method == 'POST':
        try:
            inputB = float(request.form['inputB'])
            inputD = float(request.form['inputD'])
            inputP = float(request.form['inputP'])
            inputJ = float(request.form['inputJ'])
            inputN = float(request.form['inputN'])
        except ValueError:
            return render_template(
                'optimize_form.html', 
                error="Please ensure all inputs are valid numbers."
            )

        input_params = [inputB, inputD, inputP, inputJ, inputN]
        
        best_individual, best_efficiency,v, T = optimize.optimize(input_params)
        
        return render_template(
            'optimize_result.html',
            best_individual=best_individual,
            best_efficiency=best_efficiency,
            v=v,
            T=T
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
