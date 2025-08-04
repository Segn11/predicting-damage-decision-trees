# Predicting Damage with Decision Trees

import sqlite3
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from category_encoders import OrdinalEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.validation import check_is_fitted

# Suppress FutureWarnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)

# ----------------------- DATA WRANGLING FUNCTION -----------------------
def wrangle(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # SQL query to join three tables and filter by district_id = 4
    query = """
        SELECT distinct(i.building_id) AS b_id,
               s.*,
               d.damage_grade
        FROM id_map AS i
        JOIN building_structure AS s ON i.building_id = s.building_id
        JOIN building_damage AS d ON i.building_id = d.building_id
        WHERE district_id = 4
    """

    # Load query results into a DataFrame
    df = pd.read_sql(query, conn, index_col="b_id")

    # Drop columns that leak information about the target
    drop_cols = [col for col in df.columns if "post_eq" in col]

    # Drop redundant or high-cardinality columns
    drop_cols.append("building_id")

    # Convert the last character of damage_grade ('Grade 1' → 1)
    df["damage_grade"] = df["damage_grade"].str[-1].astype(int)

    # Create a binary target column: 1 if damage is severe (> Grade 3), else 0
    df["severe_damage"] = (df["damage_grade"] > 3).astype(int)

    # Drop the original damage_grade column
    drop_cols.append("damage_grade")

    # Drop multicollinear column (correlated with others)
    drop_cols.append("count_floors_pre_eq")

    # Final drop
    df.drop(columns=drop_cols, inplace=True)

    return df


# ----------------------- LOAD AND PREPARE DATA -----------------------
df = wrangle("../nepal.sqlite")  # Provide correct path to your .sqlite file
df.head()

# Define target and features
target = "severe_damage"
X = df.drop(columns=target)
y = df[target]

# Split dataset: 80% training+validation, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Further split training set: 64% training, 16% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ----------------------- BASELINE ACCURACY -----------------------
# Baseline: always predict the most frequent class
acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))

# ----------------------- INITIAL MODEL BUILDING -----------------------
# Pipeline: OrdinalEncoder (for categoricals) + DecisionTreeClassifier
model = make_pipeline(
    OrdinalEncoder(), 
    DecisionTreeClassifier(max_depth=6, random_state=42)
)

# Train the model
model.fit(X_train, y_train)

# Evaluate model performance
acc_train = accuracy_score(y_train, model.predict(X_train))
acc_val = model.score(X_val, y_val)

print("Training Accuracy:", round(acc_train, 2))
print("Validation Accuracy:", round(acc_val, 2))

# Check the depth of the fitted decision tree
tree_depth = model.named_steps["decisiontreeclassifier"].get_depth()
print("Tree Depth:", tree_depth)

# ----------------------- HYPERPARAMETER TUNING -----------------------
# Try different max_depth values to see their effect on performance
depth_hyperparams = range(1, 50, 2)

# Lists to hold accuracy scores for each depth
training_acc = []
validation_acc = []

# Loop over different tree depths
for d in depth_hyperparams:
    # Create new model with max_depth = d
    test_model = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=d, random_state=42)    
    )
    test_model.fit(X_train, y_train)

    # Evaluate training and validation accuracy
    training_acc.append(test_model.score(X_train, y_train))
    validation_acc.append(test_model.score(X_val, y_val))

# Preview a few accuracy values
print("Training Accuracy Scores:", training_acc[:3])
print("Validation Accuracy Scores:", validation_acc[:3])

# ----------------------- PLOT ACCURACY CURVES -----------------------
plt.figure(figsize=(10, 6))
plt.plot(depth_hyperparams, training_acc, label="Training Accuracy", marker='o')
plt.plot(depth_hyperparams, validation_acc, label="Validation Accuracy", marker='s')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.title("Decision Tree Depth vs Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------- FINAL TEST SET EVALUATION -----------------------
# Evaluate the final model on unseen test data
test_acc = model.score(X_test, y_test)
print("Test Accuracy:", round(test_acc, 2))
