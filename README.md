# Customer Churn Prediction Project
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)

## Table of Contents
1. <a href="#-project-overview">✨ Project Overview</a>
2. <a href="#-key-features">🚀 Key Features</a>
3. <a href="#-project-structure">📂 Project Structure</a>
4. <a href="#-installation">🛠️ Installation</a>
5. <a href="#-usage">🧪 Usage</a>
6. <a href="#-contributing">📦 Contributing</a>
7. <a href="#-author">👤 Author</a>

## ✨ Project Overview

This end-to-end machine learning project predicts customer churn (attrition) using behavioral data and service usage patterns. The solution combines clustering for customer segmentation with classification for churn risk prediction.

## 🚀 Key Features
- Complete ML pipeline from raw data to predictions
- Production-ready modular codebase
- Comprehensive test coverage (unit + integration tests)
- Prediction API for integration with other systems

## 📂 Project Structure
```
/ML-CHURN-DE-CLIENTES
│   README.md
│   main.py                     # Main entry point
|   dataframe.py                # DataFrame used for churn study
│   .gitignore
|   
├───src
│   │   __init__.py             # Make the directory a Python package
│   │
│   ├───data
│   │       preprocessing.py    # Pre processing functions
│   │       visualization.py    # View functions
│   │
│   ├───models
│   │       clustering.py       # Clustering functions
│   │       classification.py   # Classification Model
│   │       predict.py         # Prediction functions
│   │
│   └───utils
│           helpers.py          # Auxiliary functions
│
└───tests
    │   conftest.py            # Test Settings
    │
    ├───unit
    │       test_preprocessin.py
    │       test_helpers.py
    │
    └───integration
            test_pipeline.py
```
<a id="-installation"></a>
## 🛠️ Installation

### Pre requisites
- Python 3.8+
- pip package manager

### Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/mateusgomes6/ML-customer-churn.git
cd ML-customer-churn
````
### Create and activate virtual environment:
python -m venv venv
1. On Windows:
```
venv\Scripts\activate
````
2. On macOS/Linux:
```
source venv/bin/activate
```

## 🧪 Usage

### Running the Full Pipeline
1. Pre process data:
```
python src/data/preprocessing.py
````
3. Train Model:
```
python src/models/classification.py
```
3. Generate predictions:
```
python src/models/predict.py 
````
## 📦 Contributing

We welcome contributions from the community! Please follow these steps to contribute:

1. **Fork the repository**  

2. **Create your feature branch**  
```bash
git checkout -b feature/your-feature-name
```
3. **Commit your changes**
```bash
git commit -m 'Add descriptive commit message'
````
4. **Push to the branch**
```
git push origin feature/your-feature-name
```
5. Open a Pull Request
## 👤 Author

Mateus Gomes
[GitHub](https://github.com/mateusgomes6)
[Email](mateusgomesdc@hotmail.com)
