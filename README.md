# ⚽ Football Player Overall Rating Predictor

A machine learning web application that predicts a football player's overall rating based on 29 physical, technical, and mental attributes, along with their playing position. Built using Python, Scikit-learn, Streamlit, and Plotly, the application provides interactive predictions and visual insights into player performance.

## 📌 Project Overview

The Football Player Overall Rating Predictor uses a trained Random Forest Regressor to estimate a player's overall rating on a scale of 1–99, inspired by player-rating systems used in football video games such as EA SPORTS FC and FIFA.

The project covers the complete machine learning workflow, from cleaning raw player data and engineering features to training the model and deploying it through an interactive web interface.

## ✨ Key Features

* **Overall Rating Prediction:** Predicts a player's overall rating using their attributes and pitch position.
* **Interactive Attribute Sliders:** Customize player attributes on a scale of 1–99.
* **Position-Based Prediction:** Supports 13 playing positions, including CAM, CB, CDM, CF, CM, GK, LB, LM, LW, RB, RM, RW, and ST.
* **Interactive Radar Chart:** Visualizes the player's attribute distribution using Plotly.
* **Skill Category Breakdown:** Displays a horizontal breakdown of player skill categories.
* **Rating Gauge:** Presents the predicted rating across performance tiers, from Poor to World Class.
* **Sample Player Preset:** Quickly loads a sample world-class striker profile for testing.

## 🧠 Machine Learning Workflow

### 1. Data Cleaning and Preprocessing

The project uses `messy_football_players_8000.csv`, containing an initial 8,000 player records.

The data preparation process includes:

* Removing records with invalid ages outside the range of 15–40 years.
* Handling missing numerical values using median imputation.
* Encoding categorical playing positions using One-Hot Encoding.
* Preparing the final feature set for model training.

### 2. Feature Engineering

The model uses 42 input features:

* 29 numerical player attributes covering physical, technical, and mental abilities.
* 13 One-Hot Encoded features representing playing positions.

This representation enables the model to consider both a player's individual attributes and their position on the pitch.

### 3. Model Training

A **Random Forest Regressor** from Scikit-learn is used to learn the relationship between player attributes and overall ratings.

### 4. Model Evaluation

The trained model achieved the following results:

| Evaluation Metric              |             Result |
| ------------------------------ | -----------------: |
| R² Score                       | Approximately 0.81 |
| Root Mean Squared Error (RMSE) |  Approximately 3.2 |

An R² score of approximately 0.81 indicates that the model explains around 81% of the variance in the target ratings on the evaluated dataset. An RMSE of approximately 3.2 indicates a typical prediction error magnitude of around 3.2 rating points, with larger errors having a greater influence on this metric.

## 📊 Interactive Dashboard

The Streamlit interface organizes player attributes into the following categories:

* **Attacking**
* **Defensive**
* **Physical**
* **Goalkeeping**

Users can adjust attributes, select a playing position, and generate a prediction. The resulting dashboard displays the predicted overall rating alongside visual representations of the player's abilities.

## 🛠️ Technologies Used

* **Python** — Core programming language
* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical computations
* **Scikit-learn** — Data preprocessing and Random Forest model
* **Streamlit** — Interactive web application
* **Plotly** — Interactive data visualizations
* **Joblib** — Saving and loading the trained model
* **Matplotlib and Seaborn** — Exploratory data analysis and visualization
* **Jupyter Notebook** — Data analysis and model development

## 📁 Project Structure

```text
Football-Player-Overall-Rating-Prediction/
│
├── app.py
├── notebook.ipynb
├── messy_football_players_8000.csv
└── README.md
```

*Note: The structure above represents the expected project files. Update the filenames if your repository uses different names.*

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.8 or higher
* Git
* pip

### 1. Clone the Repository

```bash
git clone https://github.com/abin11/Football-Player-Overall-Rating-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Football-Player-Overall-Rating-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows (Command Prompt):**

```bash
venv\Scripts\activate
```

**Windows (Git Bash):**

```bash
source venv/Scripts/activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

## 🚀 Future Improvements

* Experiment with additional regression algorithms and hyperparameter tuning.
* Add feature-importance analysis to explain the model's predictions.
* Expand the dataset to include more player profiles and seasons.
* Improve model performance through further feature engineering.
* Add options to compare multiple player profiles.

## 🎯 Project Objective

This project demonstrates how machine learning can be applied to football analytics by transforming player attributes into predicted overall ratings through a complete workflow involving data preprocessing, feature engineering, regression modelling, model evaluation, and interactive deployment.

## 👨‍💻 Author

**Abin K Vijayan**

GitHub: [abin11](https://github.com/abin11)

---

*Developed as a machine learning project exploring football player performance prediction and interactive data visualization.*
