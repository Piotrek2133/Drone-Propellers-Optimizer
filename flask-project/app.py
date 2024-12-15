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
            # Parse main inputs
            inputB = float(request.form['inputB'])
            inputD = float(request.form['inputD'])
            inputP = float(request.form['inputP'])
            inputJ = float(request.form['inputJ'])
            inputN = float(request.form['inputN'])
            
            # Handle initial data
            use_defaults = request.form.get('use_defaults') == 'on'
            if use_defaults:
                initial_data = [
                    (0.114, 28.24),
                    (0.134, 31.87),
                    (0.157, 32.26),
                    (0.177, 31.32),
                    (0.194, 29.65),
                    (0.208, 27.57),
                    (0.218, 25.24),
                    (0.225, 22.97),
                    (0.228, 20.94),
                    (0.228, 19.19),
                    (0.223, 17.69),
                    (0.214, 16.31),
                    (0.202, 15.07),
                    (0.185, 13.94),
                    (0.165, 12.89),
                    (0.139, 11.86),
                    (0.100, 11.04),
                    (0.060, 10.23)
                ]
            else:
                # Collect user inputs for c/R and Beta
                initial_data = []
                for i in range(18):
                    cR = float(request.form[f'cR_{i}'])
                    beta = float(request.form[f'beta_{i}'])
                    initial_data.append((cR, beta))
        except ValueError:
            return render_template('optimize_form.html', error="Please ensure all inputs are valid numbers.")

        # Perform optimization
        input_params = [inputB, inputD, inputP, inputJ, inputN]
        best_individual, best_efficiency, v, T = optimize.optimize(input_params, initial_data)
        
        return render_template(
            'optimize_result.html',
            best_individual=best_individual,
            best_efficiency=best_efficiency,
            v=v,
            T=T
        )

# @app.route('/optimize', methods=['GET', 'POST'])
# def run_optimization():
#     if request.method == 'GET':
#         return render_template('optimize_form.html')
#     elif request.method == 'POST':
#         try:
#             inputB = float(request.form['inputB'])
#             inputD = float(request.form['inputD'])
#             inputP = float(request.form['inputP'])
#             inputJ = float(request.form['inputJ'])
#             inputN = float(request.form['inputN'])

#             initial_data = []
#             for i in range(18):
#                 cR = float(request.form[f'cR_{i}'])
#                 beta = float(request.form[f'beta_{i}'])
#                 initial_data.append((cR, beta))

#         except ValueError:
#             return render_template(
#                 'optimize_form.html',
#                 error="Please ensure all inputs are valid numbers."
#             )

#         input_params = [inputB, inputD, inputP, inputJ, inputN]

#         best_individual, best_efficiency, v, T = optimize.optimize(input_params, initial_data)

#         return render_template(
#             'optimize_result.html',
#             best_individual=best_individual,
#             best_efficiency=best_efficiency,
#             v=v,
#             T=T
#         )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
