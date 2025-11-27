# 🎓 College Readiness Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Final Project for ENG Basecamp AI Training Program**  
> *Developed by David Macaulay*

An end-to-end machine learning system that predicts student college readiness based on demographic and educational background factors. This project demonstrates the complete ML pipeline from data preprocessing to model deployment via an interactive web application.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Key Learnings](#-key-learnings)
- [Future Enhancements](#-future-enhancements)
- [Technologies Used](#-technologies-used)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Project Overview

This machine learning system predicts whether a student is **college-ready** based on their demographic and educational background. College readiness is operationally defined as achieving an average score of **≥ 75** across Math, Reading, and Writing standardized assessments.

**Key Features:**
- ✅ Binary classification model (College Ready / Not Ready)
- ✅ Comparison of multiple ML algorithms (Logistic Regression & Random Forest)
- ✅ Interactive Streamlit web application
- ✅ Single and batch prediction capabilities
- ✅ Automated hyperparameter tuning with cross-validation
- ✅ Personalized recommendations based on predictions

---

## 🔍 Problem Statement

Educational institutions need data-driven tools to identify students who may need additional support to achieve college readiness. This system addresses that need by:

1. **Early Identification**: Predicting college readiness based on accessible demographic data
2. **Resource Allocation**: Helping schools target interventions effectively
3. **Equity Analysis**: Examining how demographic factors relate to educational outcomes

**Target Metric**: Binary classification predicting college readiness (average test score ≥ 75)

---

## 📊 Dataset

**Source**: Student Performance Dataset  
**Size**: 1,000 student records  
**Features**: 8 columns (5 categorical, 3 numerical)

### Features Used for Prediction:
| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Student's gender (male/female) |
| Race/Ethnicity | Categorical | Demographic group (A-E) |
| Parental Education | Categorical | Highest parental education level |
| Lunch Type | Categorical | Standard or free/reduced (socioeconomic proxy) |
| Test Prep Course | Categorical | Completion status (none/completed) |

### Target Variable:
- **College Ready**: Binary (1 = average score ≥ 75, 0 = otherwise)
- **Class Distribution**: 
  - College Ready: ~36% of students
  - Not Ready: ~64% of students

**Note**: Test scores (math, reading, writing) are used only to create the target variable and are **excluded from features** to prevent data leakage.

---

## 🛠️ Methodology

### 1. **Data Preprocessing**
- Data validation and cleaning (no missing values found)
- Target variable creation (binary college readiness indicator)
- One-hot encoding of categorical features
- Feature selection (excluding test scores from predictors)

### 2. **Model Development**
Two classification algorithms were implemented and compared:

#### **Logistic Regression**
- Pipeline: StandardScaler → Logistic Regression
- Hyperparameter tuning: Regularization strength (C), penalty type
- 5-fold cross-validation
- Best suited for interpretability

#### **Random Forest Classifier**
- Ensemble method with multiple decision trees
- Hyperparameter tuning: n_estimators, max_depth, min_samples_split
- 5-fold cross-validation
- Better for capturing non-linear relationships

### 3. **Model Selection**
- Train/test split: 80/20 with stratification
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Best model selected based on test set performance
- Model and feature columns serialized with pickle

### 4. **Deployment**
- Interactive Streamlit web application
- Real-time predictions with confidence scores
- Batch prediction via CSV upload
- Visual probability breakdowns
- Actionable recommendations

---

## 📁 Project Structure

```
College-Readiness-Prediction-System/
│
├── data/
│   └── student_performance.csv          # Raw dataset
│
├── model/
│   ├── main.py                          # ML pipeline & training script
│   ├── week6_final_project.ipynb        # Exploratory data analysis
│   ├── logistic_regression_model.pkl    # Trained LR model
│   ├── random_forest_model.pkl          # Trained RF model
│   └── feature_columns.pkl              # Feature schema
│
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
└── LICENSE                              # MIT License
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/College-Readiness-Prediction-System.git
cd College-Readiness-Prediction-System
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model
```bash
cd model
python main.py
```

This will:
- Load and preprocess the data
- Train both Logistic Regression and Random Forest models
- Perform hyperparameter tuning with cross-validation
- Save the best model and feature columns

---

## 🚀 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Single Student Prediction
1. Navigate to the main page
2. Fill in the student information form:
   - Select gender
   - Choose race/ethnicity group
   - Select parental education level
   - Choose lunch type
   - Indicate test prep completion status
3. Click **"Predict College Readiness"**
4. View prediction, confidence score, and recommendations

### Batch Prediction
1. Scroll to the **"Batch Prediction"** section
2. Upload a CSV file with the required columns:
   - `gender`
   - `race/ethnicity`
   - `parental level of education`
   - `lunch`
   - `test preparation course`
3. Click **"Run Batch Prediction"**
4. View results and download predictions as CSV

---

## 📈 Model Performance

### Best Model: [Varies by run]

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy | ~XX.X% |
| Test Set Accuracy | ~XX.X% |
| Precision (College Ready) | ~XX.X% |
| Recall (College Ready) | ~XX.X% |
| F1-Score | ~XX.X% |

**Model Comparison:**
- **Logistic Regression**: Provides interpretable coefficients showing feature importance
- **Random Forest**: Captures non-linear relationships and feature interactions

*Note: The system automatically selects the best-performing model based on test accuracy.*

---

## 💡 Key Learnings

Through completing this project as part of the ENG Basecamp AI Training Program, I gained hands-on experience in:

### **1. End-to-End ML Pipeline Development**
- Designing and implementing a complete machine learning workflow
- Understanding the importance of each stage: data collection → preprocessing → modeling → deployment
- Learning to structure ML projects for maintainability and scalability

### **2. Data Preprocessing & Feature Engineering**
- Handling categorical variables through one-hot encoding
- Creating derived target variables from existing features
- Preventing data leakage by carefully selecting features
- Understanding the impact of feature selection on model performance

### **3. Model Selection & Hyperparameter Tuning**
- Comparing multiple algorithms for the same problem
- Using GridSearchCV for systematic hyperparameter optimization
- Implementing cross-validation to prevent overfitting
- Understanding the bias-variance tradeoff in model selection

### **4. Model Evaluation**
- Going beyond accuracy to evaluate precision, recall, and F1-score
- Understanding when different metrics matter (e.g., false positives vs false negatives)
- Using stratified sampling to handle class imbalance
- Interpreting confusion matrices and classification reports

### **5. Production Deployment**
- Building interactive web applications with Streamlit
- Serializing models for reuse (pickle)
- Designing user-friendly interfaces for non-technical stakeholders
- Implementing batch prediction capabilities for scalability

### **6. Software Engineering Best Practices**
- Writing modular, reusable code
- Creating clear project structure and documentation
- Version control with Git
- Dependency management with requirements.txt

### **7. Ethical Considerations in ML**
- Recognizing potential biases in demographic-based predictions
- Understanding the social implications of automated decision systems
- Designing systems that augment rather than replace human judgment
- Considering fairness and equity in model development

### **8. Problem-Solving & Critical Thinking**
- Translating business problems into ML tasks
- Making informed decisions about model architecture
- Debugging and troubleshooting ML pipelines
- Iterating based on model performance feedback

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.8+ |
| **ML Libraries** | scikit-learn, NumPy, Pandas |
| **Web Framework** | Streamlit |
| **Data Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Pickle |
| **Development Tools** | Jupyter Notebook, Git |
| **IDE** | VS Code |

---

## 🙏 Acknowledgments

- **ENG Basecamp AI Training Program**: For providing comprehensive training and mentorship in machine learning and AI
- **Instructors & Mentors**: For guidance throughout the project development process
- **Cohort Peers**: For collaborative learning and feedback
- **Open Source Community**: For the excellent libraries and tools that made this project possible

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**David Macaulay**

*ENG Basecamp AI Training Program - Final Project*

---

## 📞 Contact & Feedback

If you have questions, suggestions, or would like to collaborate:

- 🐛 **Report Issues**: [GitHub Issues](https://github.com/master-david445/College-Readiness-Prediction-System/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/master-david445/College-Readiness-Prediction-System/discussions)

---

## 🌟 Project Status

**Status**: ✅ Completed (ENG Basecamp Final Project)  
**Last Updated**: November 2025

---

  
### ⭐ If you found this project helpful, please consider giving it a star!

**Built with ❤️ as part of ENG Basecamp AI Training Program**
