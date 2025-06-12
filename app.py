from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

# Load saved model components
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
feature_list = joblib.load("model_features.pkl")  # column order from training

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/getstarted')
def get_started():
    return render_template("getstarted.html")

@app.route('/churn')
def churn_page():
    return render_template("churn.html")

@app.route('/predict_by_input', methods=["POST"])
def predict_by_input():
    try:
        # Get form inputs
        gender = request.form['gender']
        geography = request.form['geography']
        marital_status = request.form['marital_status']
        age = float(request.form['age'])
        tenure = float(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = float(request.form['num_of_products'])
        estimated_salary = float(request.form['estimated_salary'])
        credit_score = float(request.form['credit_score'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])

        # Engineering
        monthly_income = estimated_salary
        emi_amount = balance / 12 if balance > 0 else 0
        sentiment_score = 0.5  # Placeholder, or calculate if needed

        # Combine categorical data as trained
        combined_cats = f"{gender} {geography} Chennai {marital_status} Savings Yes 101 Happy"
        tfidf_features = tfidf.transform([combined_cats])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf.get_feature_names_out())

        # Numeric Data
        numeric_data = [[
            age, tenure, balance, num_of_products, estimated_salary,
            monthly_income, credit_score, emi_amount, sentiment_score
        ]]
        numeric_df = pd.DataFrame(scaler.transform(numeric_data), columns=[
            'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
            'Monthly_Income', 'CreditScore', 'EMI_Amount', 'Sentiment_Score'
        ])

        # Combine all inputs
        final_input = pd.concat([numeric_df, tfidf_df], axis=1)

        # Align columns with training set
        for col in feature_list:
            if col not in final_input:
                final_input[col] = 0  # fill missing with 0
        final_input = final_input[feature_list]  # ensure correct order

        # Predict
        prediction = model.predict(final_input)[0]
        label = "🟠 Customer is likely to churn" if prediction == 1 else "🟢 Customer is likely to stay"
        return render_template("result.html", result=label)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
