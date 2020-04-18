# Machine-Learning Pipeline (BRAF)

## Purpose: Build and test Biased Random Forest Classifier
### Requirements: 
    1) Train model on the dataset with specified hyperparameters
      - K = 10, p = 0.5, s = 100
    2) Print out precision, recall, AUPRC, and AUROC for 10-fold cross-validation on the training set and test set
    3) Create ROC and PRC curves for both cross-validation and test set results
    
#### Folder Structure
    - data_plit/
        - main.py
    - braf/
        - algorithms/
            - NearestNeighbors.py
            - RandomForest.py
        - main.py
    - metrics/
        - auc.py
        - average_precision_score.py
        - binary_clf_curve.py
        - precision_recall_curve.py
        - roc_curve.py
    - plots/
        - main.py
    - Data/
    - .gitignore
    - main.py
    - README.md
    - requirements.txt
    
#### Pipeline and Model Logic
    1) Run data
    2) run model pipeline under two conditions
        a) train_test_split: train = 0.8, test = 0.2
        b) k_fold_crossvalidation: train = 0.9, test = 0.1
    3) Model Pipeline Logic
        a) Split test data into majority and minority labels
        b) Build critical dataset by running dataset of minority labels
            1) run each minority label row to find the Nearest Neighbors
            2) append each nearest neighor output to critical dataset
            3) remove duplicates
        c) Build BRAF
            1) run RandomForest on full dataset s * (1 - p)
            2) run RandomForest on critical dataset s * p
            3) Combine the two RandomForest Estimators
    4) Evaluation Report on the two data split conditions
    
#### Instructions
    pip install -r requirements.txt
    python main.py
 
