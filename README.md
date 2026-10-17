# Finance ML Suite

**A Unified Machine Learning Platform for Financial Prediction and Risk Analysis**

Finance ML Suite is a modular machine learning project designed to bring multiple financial prediction and analysis algorithms together in a single application. It uses Python, Scikit-learn, and FastAPI to build a reusable backend for financial forecasting, fraud detection, loan-default prediction, and exploratory data analysis.

The project follows a modular architecture in which data preprocessing, feature engineering, model training, model evaluation, model persistence, and API-based inference are handled separately.

## Key Features

- **Loan Amount Prediction:** Predicts loan amounts using regression models.
- **Fraud Detection:** Classifies financial records to identify potentially fraudulent activity.
- **Loan Default Prediction:** Estimates whether a borrower may default on a loan.
- **Data Preprocessing:** Handles missing values, duplicates, invalid records, and data validation.
- **Feature Engineering:** Creates meaningful financial features and ratios from raw data.
- **Feature Selection:** Evaluates features using statistical and model-based methods.
- **Multiple ML Algorithms:** Supports a modular approach to implementing and comparing different algorithms.
- **Model Persistence:** Saves trained models for reuse without retraining on every prediction.
- **REST API:** Exposes prediction functionality through a FastAPI backend.
- **Interactive API Documentation:** Enables endpoint testing through Swagger UI.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine learning | Scikit-learn |
| Data processing | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| API framework | FastAPI |
| API server | Uvicorn |
| Model serialization | Joblib |
| Model evaluation | Scikit-learn metrics |
| Development and experimentation | Jupyter Notebook / Google Colab |
| Version control | Git and GitHub |

## Project Architecture

The project separates data processing, training scripts, model artifacts, and API endpoints to make the codebase easier to maintain and extend.

```text
FinanceML/
├── app/
│   ├── main.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── generate_data.py
│   │   └── loans.csv
│   │
│   ├── ml/
│   │   ├── train_linear_regression.py
│   │   ├── train_logistic_regression.py
│   │   ├── train_decision.py
│   │   ├── train_random_forest.py
│   │   ├── train_gradient_boosting.py
│   │   ├── train_svm.py
│   │   └── ...
│   │
│   ├── models/
│   │   ├── linear_regression.pkl
│   │   ├── logistic_regression.pkl
│   │   └── ...
│   │
│   ├── routers/
│   │   └── ...
│   │
│   └── schemas/
│       └── loan_application.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The tree illustrates the intended organization. Retain the actual filenames and directories present in your repository; not every example file is necessarily implemented.*

### Main Components

- `app/main.py`: Initializes the FastAPI application and registers available routers.
- `app/core/config.py`: Centralizes shared configuration, data paths, model paths, and feature definitions.
- `app/data/generate_data.py`: Handles dataset generation or preparation, inspection, validation, cleaning, feature engineering, encoding, and dataset splitting as configured.
- `app/ml/`: Contains individual training modules for different algorithms.
- `app/models/`: Stores serialized models and any required preprocessing artifacts.
- `app/routers/`: Defines API endpoints for prediction and other supported operations.
- `app/schemas/`: Defines request and response validation schemas.

## Machine Learning Tasks

Finance ML Suite covers three primary prediction targets.

| Task | Target variable | Problem type |
|---|---|---|
| Loan amount prediction | `loan_amount` | Regression |
| Fraud detection | `is_fraud` | Binary classification |
| Loan default prediction | `default` | Binary classification |

The dataset and its labels must be defined consistently with these tasks. In particular, fraud detection and loan-default prediction are separate classification problems and should be trained and evaluated independently.

## Data Processing Pipeline

The data pipeline prepares raw financial data for training and inference.

1. **Data loading:** Load the source dataset into a Pandas DataFrame.
2. **Data inspection:** Examine column types, distributions, missing values, duplicates, and target distributions.
3. **Data validation:** Check invalid or inconsistent records and enforce domain-specific constraints.
4. **Data cleaning:** Handle missing values, duplicates, and invalid entries according to the preprocessing rules.
5. **Feature engineering:** Construct financial ratios and other derived variables.
6. **Encoding:** Convert categorical variables into numerical representations.
7. **Feature selection:** Identify relevant predictors using statistical and model-based techniques.
8. **Dataset splitting:** Separate features and targets into training and testing datasets.
9. **Feature scaling:** Fit the required scaler on the training data and apply the same transformation to the test data.
10. **Model training:** Train the selected algorithm using the prepared training data.
11. **Evaluation:** Measure predictive performance on held-out test data.
12. **Model persistence:** Save the trained model and any required preprocessing objects for inference.

### Example Engineered Features

Depending on the available source columns, useful financial features can include:

- Debt-to-income ratio
- Spending-to-income ratio
- Savings-to-income ratio
- Debt-to-savings ratio
- Credit utilization
- Salary-to-loan ratio
- Online transaction ratio
- Average transaction amount

These features provide additional information about borrower financial behavior and transaction patterns. Their usefulness should be established through validation and evaluation rather than assumed in advance.

## Algorithm Development

The suite is designed to support multiple machine learning algorithms through separate training modules.

| Category | Algorithms |
|---|---|
| Regression | Linear Regression |
| Classification | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes |
| Clustering | K-Means, Hierarchical Clustering, DBSCAN |
| Dimensionality reduction | Principal Component Analysis (PCA) |

The list represents the algorithm roadmap; implementation and evaluation status may differ by module.

Each supervised learning module should follow a consistent workflow:

- Load the prepared training and testing datasets.
- Select the target-specific feature set.
- Apply the required preprocessing.
- Train the model.
- Generate predictions.
- Calculate evaluation metrics.
- Save the model and its required preprocessing artifacts.
- Integrate the model into the API when inference support is implemented.

## Feature Selection Strategy

Feature selection is used to reduce irrelevant or redundant predictors and improve model performance and interpretability.

The project can compare several feature-selection approaches:

- Correlation analysis
- Mutual information
- Recursive Feature Elimination (RFE)
- Recursive Feature Elimination with Cross-Validation (RFECV)
- L1 regularization
- Random Forest feature importance

The selected features should be evaluated separately for each prediction target. A feature that is useful for fraud detection may not be equally useful for predicting loan defaults or loan amounts.

Feature selection should be performed using training data only, with appropriate cross-validation, to avoid information leakage from the test set.

## Model Evaluation

Different problem types require different evaluation metrics.

### Regression

Recommended metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (\(R^2\))

### Classification

Recommended metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, where appropriate

Fraud detection and loan default may involve imbalanced classes. Therefore, accuracy alone is insufficient to establish model quality. Precision, recall, F1-score, and the confusion matrix should also be examined.

Threshold selection should reflect the cost of false positives and false negatives in each financial use case.

### Clustering

Recommended metrics, where applicable:

- Silhouette score
- Davies–Bouldin index
- Calinski–Harabasz score

Clustering results should also be interpreted in terms of the financial patterns represented by the discovered groups.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by your installed dependencies
- pip
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd FinanceML
```

Replace `<YOUR_REPOSITORY_URL>` with the actual Git repository URL.

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If `requirements.txt` is available and up to date:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies used by your implemented modules and record the tested versions in `requirements.txt`.

### 4. Prepare the Dataset

Run the data preparation module from the repository root if it is configured to generate or prepare the dataset:

```bash
python -m app.data.generate_data
```

Verify that the processed datasets and target columns are available at the paths configured in `app/core/config.py`.

### 5. Train Models

Run the training modules corresponding to the algorithms you want to train.

For example:

```bash
python -m app.ml.train_linear_regression
python -m app.ml.train_logistic_regression
```

Use the actual module names in your repository for other algorithms.

Confirm that each training script completes successfully and saves the expected model artifacts before using the corresponding prediction endpoint.

### 6. Start the FastAPI Server

Run the application from the repository root:

```bash
uvicorn app.main:app --reload
```

The development server will generally be available at:

```text
http://127.0.0.1:8000
```

### 7. Open API Documentation

Visit:

```text
http://127.0.0.1:8000/docs
```

Swagger UI allows you to inspect available endpoints, submit prediction requests, and review the API responses.

## API Usage

The application is designed to expose prediction capabilities through REST endpoints. The exact routes depend on which routers have been registered in `app/main.py`.

Previously defined endpoint examples include:

| Endpoint | Purpose |
|---|---|
| `/linear-regression/predict` | Loan amount prediction using the linear regression model |
| `/logistic-regression/predict` | Prediction using the logistic regression module |

Check the running application's Swagger documentation for the current endpoint paths, HTTP methods, request schemas, and response formats.

### Example API Request

The following is a conceptual example; adapt the endpoint and fields to the request schema implemented in your application.

```http
POST /linear-regression/predict
Content-Type: application/json
```

```json
{
  "monthly_income": 50000,
  "credit_score": 720,
  "existing_debt": 10000
}
```

The actual request must include every required field defined by the relevant Pydantic schema.

## Model Persistence and Inference

Trained models can be serialized using Joblib and loaded for subsequent predictions.

```python
import joblib

joblib.dump(model, "app/models/model.pkl")

loaded_model = joblib.load("app/models/model.pkl")
predictions = loaded_model.predict(X)
```

This is a simplified illustration; use the paths and loading conventions defined in your training and inference modules.

If a model requires feature scaling, encoding, or other transformations, inference must apply the same fitted preprocessing objects used during training. Do not fit a new scaler on individual prediction requests.

Model artifacts should be versioned alongside the corresponding feature definitions and preprocessing configuration.

## Project Development Phases

The project follows a phased development approach.

| Phase | Objective |
|---|---|
| Phase 1 | Define the architecture and initial dataset |
| Phase 2 | Data inspection, validation, cleaning, feature engineering, encoding, and dataset preparation |
| Phase 3 | Implement, train, evaluate, and persist machine learning models |
| Phase 4 | Perform target-specific feature selection and model comparison |
| Phase 5 | Integrate supported models with FastAPI prediction endpoints |
| Phase 6 | Testing, documentation, optimization, and deployment |

These phases describe the development workflow; individual tasks may be completed in parallel or revisited as evaluation results become available.

## Testing and Quality Checks

Before integrating a model into the API, verify that:

- Training and testing datasets have the expected shapes.
- Target columns are not accidentally included among the predictors.
- Preprocessing does not leak information from the test set.
- Classification metrics include positive-class performance.
- Model and preprocessing artifacts can be loaded successfully.
- Predictions use the expected feature order and data types.
- API requests are validated and invalid inputs return appropriate errors.
- Training and inference use consistent feature transformations.

## Limitations

- Predictive performance depends on the quality and representativeness of the training dataset.
- Synthetic data may not capture real-world financial behavior or fraud patterns.
- Fraud and default predictions are risk estimates, not definitive determinations.
- Model outputs should not be treated as the sole basis for consequential financial decisions.
- Production deployment requires additional validation, monitoring, security controls, and appropriate privacy safeguards.

## Future Enhancements

Potential improvements include:

- Automated comparison of algorithms across consistent train/test splits.
- Cross-validation and hyperparameter tuning.
- Target-specific feature selection and explainability using SHAP.
- Prediction confidence and calibrated probability estimates where appropriate.
- Model versioning and experiment tracking.
- Automated data and model drift monitoring.
- Unit, integration, and API tests.
- Docker-based deployment and CI/CD.
- A frontend dashboard for predictions and model evaluation.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement the change and add relevant tests.
4. Verify that existing modules continue to work.
5. Submit a pull request describing the changes.

## License

No license has been specified yet. Add a `LICENSE` file and update this section before distributing the project under a particular license.

---

**Finance ML Suite** — A modular machine learning platform for financial prediction, fraud detection, and risk analysis.
