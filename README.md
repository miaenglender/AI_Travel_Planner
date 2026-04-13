# 🏀 March Madness Predictor (AI Course Project)

## 👤 Author
Mia Englender

---

## 📌 Problem Statement

Predicting the outcome of NCAA basketball games is a complex problem involving team strength, historical performance, and game-specific conditions.

This project applies machine learning to predict the winner between two NCAA teams using historical game data and engineered performance features.

The goal is to build an interactive system that can estimate win probability between teams using both statistical modeling and real-world sports data.

---

## 🧠 Approach

This project applies a supervised machine learning approach using **Logistic Regression** to classify game outcomes.

We engineer features from historical NCAA data and optionally enhance predictions using live team statistics.

### Key idea:
Transform raw game history into meaningful predictive features for a binary classification model.

---

## 📊 Dataset

We use the Kaggle competition dataset:

- [March Machine Learning Mania 2024](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data)

### Data includes:
- Regular season game results
- Tournament results
- Team identifiers
- Team seeds
- Historical performance statistics

---

## ⚙️ Feature Engineering

The model is trained using the following features:

### 🏀 Historical Features
- Seed difference between teams
- Elo rating (margin-based strength estimation)
- Win percentage
- Average scoring margin

### 🔥 Optional Live Features
- Recent win percentage
- Offensive rating (if available)
- Defensive rating (if available)

---

## 🤖 Model

We use:

- Logistic Regression (scikit-learn)
- Train/test split evaluation
- Probability-based prediction output

The model outputs the probability that Team A wins the matchup.

---

## 🧪 System Pipeline

1. Load NCAA historical dataset
2. Compute team-level statistics (Elo, win %, margin)
3. Build training dataset from game outcomes
4. Train Logistic Regression model
5. Deploy interactive prediction app using Streamlit

---

## 🖥️ Application

The system is deployed as an interactive web app using Streamlit.

### Features:
- Select two NCAA teams
- Predict winner instantly
- Display win probability
- Show team statistics comparison
- Optional live data integration

---

## 📈 Results

The model is evaluated using a train/test split approach.

- Uses classification accuracy as baseline metric
- Produces probabilistic outputs for interpretability
- Demonstrates reasonable predictive power using engineered features

---

## ⚠️ Limitations

- Uses historical data which may not fully reflect current team performance
- Live data depends on external availability and consistency
- Model is relatively simple (Logistic Regression)

---

## 🔮 Future Work

- Use more advanced models (XGBoost, Neural Networks)
- Improve live data integration
- Add tournament simulation (bracket prediction)
- Feature importance visualization
- Expand dataset with real-time sports APIs

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run src/app.py
