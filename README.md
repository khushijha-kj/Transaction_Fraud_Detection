### Transaction Fraud Detection

This project provides an end-to-end look at detecting fraudulent financial transactions using machine learning. It covers the entire workflow from initial data exploration to training and evaluating a predictive model.



### About the Project

The core problem for financial companies is a constant balancing act: catching fraudulent transactions without accidentally blocking legitimate customer payments. Missing a fraud costs money, but false alarms can damage customer trust.

This project was built with a clear goal in mind: create a model that prioritizes catching as many frauds as possible (high **Recall**) while also keeping the false alarms low (good **Precision**). The final output is a trained model and a detailed analysis of its performance.

-----

### The Dataset

  * **Location:** `dataset/Fraud.csv`
  * **Size:** Over 6 million rows and 11 columns of transaction data.
  * **Key points:** There are no missing values, but a significant challenge is the extreme **class imbalance**—frauds make up a tiny fraction of the total transactions.

-----

### How it Works

The project follows a standard data science process.

#### 1\. Data Exploration

  * **Initial Analysis:** We first looked at transaction types and found that only `TRANSFER` and `CASH_OUT` transactions showed any fraudulent activity.
  * **Creating Features:** Two new features, `balanceDiffOrig` and `balanceDiffDest`, were engineered to better represent changes in account balances.
  * **Feature Selection:** We decided to drop some columns like `nameOrig` and `nameDest` since they weren't needed for the model.

#### 2\. Modeling

  * **Model Choice:** A **Logistic Regression** classifier was chosen and built into a `scikit-learn` pipeline.
  * **Data Preparation:** Before training, numeric features were scaled and the transaction type was converted into a format the model could understand.
  * **Training:** The data was split into training (70%) and testing (30%) sets.

-----

### Results

The final trained model is saved as `fraud_detection_model.pkl`. This single file contains all the necessary steps, including data preprocessing, so it's easy to use.

  * **F1 Score:** The model achieved an F1-score of around **95%** on the test data.
  * **Evaluation:** A detailed classification report was generated to show key metrics like precision and recall for both fraud and non-fraud transactions.

-----

### Getting Started

You have two options for running the project.

#### Run Locally

1.  **Dependencies:** Make sure you have **Python 3.8+** and a notebook environment like Jupyter.
2.  **Clone:** Get a copy of the project from the repository.
    ```bash
    git clone https://github.com/khushijha-kj/Transaction_Fraud_Detection.git
    cd Transaction_Fraud_Detection
    ```
3.  **Setup:** Create a virtual environment and install the required libraries.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```
4.  **Run:** Place the `Fraud.csv` file inside a `dataset` folder in the project directory and run the notebook to generate the model and results.

-----
