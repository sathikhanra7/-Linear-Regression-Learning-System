from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
CORS(app)

# Optional: limit upload size (e.g., 10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {'csv'}

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Uploaded file is too large. Maximum size is 10MB.'}), 413

# Global variable to store the current dataset
current_dataset = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_plot():
    """Helper function to create base64 encoded plots"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{image_base64}"

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    global current_dataset
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            # Read CSV file into pandas DataFrame
            df = pd.read_csv(file)
            current_dataset = df.copy()

            # Prepare preview data
            head = df.head().to_dict(orient='records')
            shape = df.shape
            dtypes = df.dtypes.apply(lambda x: x.name).to_dict()

            response = {
                'head': head,
                'shape': {'rows': shape[0], 'columns': shape[1]},
                'dtypes': dtypes,
                'columns': list(df.columns)
            }
            return jsonify(response), 200

        except Exception as e:
            return jsonify({'error': f'Failed to process CSV file: {str(e)}'}), 400
    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    global current_dataset
    if current_dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    data = request.get_json()
    df = current_dataset.copy()

    preprocessing_steps = []

    # Handle missing values
    if data.get('handle_missing'):
        method = data.get('missing_method', 'mean')
        if method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            preprocessing_steps.append(f"Filled missing values in numeric columns with mean")
        elif method == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            preprocessing_steps.append(f"Filled missing values in numeric columns with median")
        elif method == 'drop':
            df = df.dropna()
            preprocessing_steps.append("Dropped rows with missing values")

    # Feature scaling
    if data.get('scaling'):
        scaler_type = data.get('scaler', 'standard')
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if scaler_type == 'standard':
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            preprocessing_steps.append("Applied StandardScaler to numeric features")
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            preprocessing_steps.append("Applied MinMaxScaler to numeric features")

    current_dataset = df

    return jsonify({
        'message': 'Preprocessing completed',
        'steps': preprocessing_steps,
        'shape': {'rows': df.shape[0], 'columns': df.shape[1]},
        'head': df.head().to_dict(orient='records'),
        'columns': list(df.columns),
        'dtypes': df.dtypes.apply(lambda x: x.name).to_dict()
    })

@app.route('/reset-dataset', methods=['POST'])
def reset_dataset():
    global current_dataset
    current_dataset = None
    return jsonify({'message': 'Dataset reset successfully'}), 200

@app.route('/eda/correlation', methods=['GET'])
def get_correlation():
    global current_dataset
    if current_dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    # Calculate correlation matrix for numeric columns
    numeric_df = current_dataset.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    correlation_plot = create_plot()

    return jsonify({
        'correlation_matrix': correlation_matrix.to_dict(),
        'plot': correlation_plot
    })

@app.route('/eda/distributions', methods=['GET'])
def get_distributions():
    global current_dataset
    if current_dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns
    plots = {}

    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(current_dataset[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plots[col] = create_plot()

    return jsonify({
        'distributions': plots,
        'numeric_columns': list(numeric_cols)
    })

@app.route('/eda/scatter-plots', methods=['POST'])
def get_scatter_plots():
    global current_dataset
    if current_dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    data = request.get_json()
    target_col = data.get('target')
    feature_cols = data.get('features', [])

    if not target_col or not feature_cols:
        return jsonify({'error': 'Target and feature columns required'}), 400

    plots = {}

    for feature in feature_cols:
        if feature in current_dataset.columns and target_col in current_dataset.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=current_dataset, x=feature, y=target_col)
            plt.title(f'{feature} vs {target_col}')
            plt.xlabel(feature)
            plt.ylabel(target_col)
            plots[f'{feature}_vs_{target_col}'] = create_plot()

    return jsonify({'scatter_plots': plots})

@app.route('/train-model', methods=['POST'])
def train_model():
    global current_dataset
    if current_dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    data = request.get_json()
    target_col = data.get('target')
    feature_cols = data.get('features', [])
    test_size = data.get('test_size', 0.2)
    use_cv = data.get('use_cv', False)
    cv_folds = data.get('cv_folds', 5)

    if not target_col or not feature_cols:
        return jsonify({'error': 'Target and feature columns required'}), 400

    missing_cols = [col for col in [target_col] + feature_cols if col not in current_dataset.columns]
    if missing_cols:
        return jsonify({'error': f'Missing selected columns in dataset: {", ".join(missing_cols)}'}), 400

    numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [col for col in [target_col] + feature_cols if col not in numeric_cols]
    if non_numeric:
        return jsonify({'error': f'Selected columns must be numeric: {", ".join(non_numeric)}'}), 400

    try:
        # Prepare data
        X = current_dataset[feature_cols]
        y = current_dataset[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Cross-validation if requested
    cv_scores = None
    if use_cv:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

    # Create visualization for single feature
    if len(feature_cols) == 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred_test, color='red', label='Predicted')
        plt.xlabel(feature_cols[0])
        plt.ylabel(target_col)
        plt.title('Actual vs Predicted Values')
        plt.legend()
        regression_plot = create_plot()
    else:
        regression_plot = None

    return jsonify({
        'coefficients': dict(zip(feature_cols, model.coef_)),
        'intercept': model.intercept_,
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
        'regression_plot': regression_plot,
        'feature_count': len(feature_cols)
    })

@app.route('/predict', methods=['POST'])
def make_prediction():
    global current_dataset
    if current_dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400

    data = request.get_json()
    target_col = data.get('target')
    feature_cols = data.get('features', [])
    input_values = data.get('input_values', {})

    if not target_col or not feature_cols:
        return jsonify({'error': 'Target and feature columns required'}), 400

    missing_cols = [col for col in [target_col] + feature_cols if col not in current_dataset.columns]
    if missing_cols:
        return jsonify({'error': f'Missing selected columns in dataset: {", ".join(missing_cols)}'}), 400

    numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [col for col in [target_col] + feature_cols if col not in numeric_cols]
    if non_numeric:
        return jsonify({'error': f'Selected columns must be numeric: {", ".join(non_numeric)}'}), 400

    try:
        # Prepare training data
        X = current_dataset[feature_cols]
        y = current_dataset[target_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Prepare input for prediction
        input_df = pd.DataFrame([input_values])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Calculate step-by-step
        steps = []
        result = model.intercept_
        steps.append(f"Start with intercept: {model.intercept_:.4f}")

        for i, feature in enumerate(feature_cols):
            coeff = model.coef_[i]
            value = input_values.get(feature, 0)
            contribution = coeff * value
            result += contribution
            steps.append(f"{feature} ({value}) * {coeff:.4f} = {contribution:.4f}")

        steps.append(f"Final prediction: {result:.4f}")

        return jsonify({
            'prediction': prediction,
            'steps': steps,
            'input_values': input_values,
            'coefficients': dict(zip(feature_cols, model.coef_)),
            'intercept': model.intercept_
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/explain-linear-regression', methods=['GET'])
def explain_linear_regression():
    explanation = {
        'simple_linear': {
            'hypothesis': 'h(x) = θ₀ + θ₁x',
            'cost_function': 'J(θ) = (1/2m) Σ(h(xⁱ) - yⁱ)²',
            'gradient_descent': 'θⱼ := θⱼ - α * ∂J/∂θⱼ',
            'description': 'Simple linear regression finds the best straight line that fits the data points.'
        },
        'multiple_linear': {
            'hypothesis': 'h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ',
            'description': 'Multiple linear regression extends simple regression to multiple features.'
        },
        'learning_steps': [
            '1. Initialize parameters (θ₀, θ₁, ..., θₙ)',
            '2. Calculate hypothesis for each training example',
            '3. Compute cost function (how well the model fits)',
            '4. Update parameters using gradient descent',
            '5. Repeat until convergence'
        ]
    }
    return jsonify(explanation)

if __name__ == '__main__':
    app.run(debug=True)