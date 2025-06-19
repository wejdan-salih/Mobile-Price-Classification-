 

📖 Project Overview

This repository contains an end-to-end machine learning project focused on predicting the price range of mobile phones based on their hardware specifications. The project leverages a well-structured dataset from Kaggle and implements a complete ML pipeline, from data cleaning and exploratory data analysis (EDA) to model training, evaluation, and comparison.

The primary objective is to build and compare the performance of two distinct supervised learning classifiers—Decision Tree and K-Nearest Neighbors (KNN)—to determine the most effective model for this classification task.

📊 Dataset

The project utilizes the Mobile Price Classification dataset, sourced from Kaggle. It is an ideal dataset for practicing classification tasks due to its clean structure and balanced class distribution.

Link to Dataset: Mobile Price Classification on Kaggle

Key Characteristics:

Instances: 2000

Features: 21 (including battery_power, ram, int_memory, screen dimensions, etc.)

Target Variable: price_range, categorized into four balanced classes:

0: Low Cost

1: Medium Cost

2: High Cost

3: Very High Cost

The dataset is perfectly balanced, with 500 samples in each class, which ensures that our models are trained without bias and allows for the use of accuracy as a reliable primary metric.

![alt text](path/to/your/class_distribution_plot.png)

(You would save the class distribution plot from your notebook as a PNG and add it here)

🚀 Project Pipeline

The project follows a systematic machine learning workflow:

1. Data Cleaning and Preprocessing

Null & Duplicate Check: The dataset was initially inspected and found to be free of missing values and duplicate records.

Handling Invalid Zeros: Certain features like px_height (pixel height) and sc_w (screen width) contained invalid zero values. These were imputed with the median of their respective columns to maintain data integrity.

Exploratory Data Analysis (EDA): A comprehensive EDA was performed to understand feature relationships. A correlation heatmap revealed a strong positive correlation between price_range and ram, identifying RAM as a key predictor.

![alt text](path/to/your/correlation_heatmap.png)

(Save the correlation heatmap from your notebook and add it here)

2. Feature Scaling & Data Splitting

Feature Scaling: StandardScaler was applied to all features. This step is crucial for distance-based algorithms like KNN, which are sensitive to the magnitude of feature values.

Train-Test Split: The dataset was split into training (80%) and testing (20%) sets. A stratified split was used to ensure the class proportions were maintained in both subsets, leading to more reliable model evaluation.

3. Model Training & Evaluation

Two distinct classification models were trained and evaluated to provide a comparative analysis.

Model 1: Decision Tree Classifier

Rationale: Chosen for its interpretability, ability to handle non-linear relationships, and robustness to outliers and unscaled data.

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

Model 2: K-Nearest Neighbors (KNN) Classifier

Rationale: Chosen as a contrasting, instance-based model that relies on distance metrics. It serves as an excellent baseline and highlights the importance of feature scaling.

Evaluation Metrics: Same as the Decision Tree for a fair comparison.

📈 Results and Analysis

The models were evaluated on the held-out test set (400 samples). The Decision Tree classifier significantly outperformed the KNN model.

Model	Accuracy	F1-Score (Weighted Avg)
Decision Tree	82.25%	0.82
K-Nearest Neighbors (k=5)	52.50%	0.53
Confusion Matrix Analysis
<p align="center">
<img src="path/to/your/decision_tree_cm.png" width="45%" alt="Decision Tree Confusion Matrix" />
<img src="path/to/your/knn_cm.png" width="45%" alt="KNN Confusion Matrix" />
</p>


The Decision Tree's confusion matrix shows a strong diagonal, indicating high accuracy across all four price classes with minimal misclassifications.

The KNN's matrix reveals significant confusion, especially between the middle classes (1 and 2), confirming its struggle with the high-dimensional feature space.

Key Takeaway

The Decision Tree is the superior model for this problem. Its inherent ability to create non-linear decision boundaries and partition the feature space based on key predictors like ram makes it highly effective. In contrast, KNN's performance was hampered by the "curse of dimensionality" and overlapping class distributions.

💡 Future Improvements

This project provides a solid foundation. Future work could include:

Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to find the optimal hyperparameters for both the Decision Tree (e.g., max_depth, min_samples_split) and KNN (e.g., n_neighbors).

Advanced Models: Implement and evaluate more sophisticated ensemble models like Random Forest, Gradient Boosting (XGBoost), or Support Vector Machines (SVM), which often yield higher accuracy.

Feature Engineering: Explore the creation of new features, such as screen area (sc_h * sc_w) or pixel density, to potentially improve model performance.

⚙️ How to Run

To replicate this project on your local machine, follow these steps:

Clone the repository:

Generated bash
git clone https://github.com/your-username/mobile-price-classification.git
cd mobile-price-classification


Set up a virtual environment (recommended):

Generated bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install the required dependencies:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt in your environment.)

Launch Jupyter Notebook:

Generated bash
jupyter notebook
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Open and run the assignment.ML.wejdanAIU.ipynb notebook. Ensure the train.csv file is in the same directory.
