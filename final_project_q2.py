'''
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

training_code_set = pd.read_csv("Training_Code_Set.csv")
test_code_set = pd.read_csv("Test_Code_Set.csv")
training_set = pd.read_csv("Training_Set.csv")
test_set = pd.read_csv("Test_Set.csv")


x_train = training_set.iloc[:, :-1]
y_train = training_set.iloc[:, -1]

x_test = test_set.iloc[:, :-1]
y_test = test_set.iloc[:, -1]

import sklearn.cluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load datasets
training_set = pd.read_csv("Training_Set.csv")
testing_set = pd.read_csv("Test_Set.csv")

# Prepare training data
training_data = training_set.drop(columns=['TYPE'])
diagnoses = training_set['TYPE'].unique()
models = []
centroids = []

# Train KMeans for each diagnosis
for diagnosis in diagnoses:
    ind_diagnosis_set = training_set[training_set['TYPE'] == diagnosis]
    red_ind_diagnosis_set = ind_diagnosis_set.drop(columns=['TYPE']).values
    
    # Use one cluster for each diagnosis
    kmean = sklearn.cluster.KMeans(
        n_clusters=1, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0,
        random_state=None, copy_x=True, algorithm='lloyd'
    ).fit(red_ind_diagnosis_set)
    
    models.append(kmean)
    centroids.append(kmean.cluster_centers_[0])  # Only one cluster center per diagnosis
    
    # Plot clusters for visualization
    plt.scatter(red_ind_diagnosis_set[:, 0], red_ind_diagnosis_set[:, 1], s=10, label=f"{diagnosis} Points")
    plt.scatter(kmean.cluster_centers_[0, 0], kmean.cluster_centers_[0, 1], s=200, marker='x', c='red', label=f"{diagnosis} Centroid")

plt.title('Diagnoses Clusters')
plt.legend()
plt.show()

# Prepare test data
test_data = testing_set.drop(columns=['TYPE'])
test_labels = testing_set['TYPE'].values
test_data = test_data.values  # Convert to numpy array

# Predict the closest cluster for each test point with constraints
predicted_labels = []
for idx, test_point in enumerate(test_data):
    distances = [np.linalg.norm(test_point - centroid) for centroid in centroids]
    predicted_label = diagnoses[np.argmin(distances)]  # Choose the label of the closest centroid
    
    # Apply constraints: Rule-based reclassification
    if testing_set.iloc[idx]['FEVER'] == 1:
        if predicted_label == "ALLERGY":
            predicted_label = "FLU"  # Fever implies it's not likely to be Allergy
        elif predicted_label == "COLD":
            predicted_label = "FLU"  # Fever often rules out common cold

    predicted_labels.append(predicted_label)

# Calculate accuracy and misclassification breakdown for each diagnosis
accuracy_details = {}
for true_label, predicted_label in zip(test_labels, predicted_labels):
    if true_label not in accuracy_details:
        accuracy_details[true_label] = {"correct": 0, "total": 0, "errors": {diag: 0 for diag in diagnoses}}
    accuracy_details[true_label]["total"] += 1
    if true_label == predicted_label:
        accuracy_details[true_label]["correct"] += 1
    else:
        accuracy_details[true_label]["errors"][predicted_label] += 1

# Print accuracy and misclassification breakdown for each diagnosis
for diagnosis, details in accuracy_details.items():
    total = details["total"]
    correct = details["correct"]
    correct_proportion = correct / total if total > 0 else 0
    print(f"Diagnosis: {diagnosis}")
    print(f"  Total: {total}")
    print(f"  Correct: {correct} ({correct_proportion:.2%})")
    print(f"  Misclassifications:")
    for misdiagnosis, count in details["errors"].items():
        if count > 0:
            proportion = count / (total - correct) if total - correct > 0 else 0
            print(f"    Misclassified as {misdiagnosis}: {count} ({proportion:.2%})")

# Calculate and print overall accuracy
overall_accuracy = np.mean(np.array(predicted_labels) == test_labels)
print(f"\nOverall Clustering-based Prediction Accuracy: {overall_accuracy:.4f}")
'''

import sklearn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Reload the training and testing sets
training_set_path = 'Training_Set.csv'
testing_set_path = 'Test_Set.csv'

training_set = pd.read_csv(training_set_path)
testing_set = pd.read_csv(testing_set_path)

# Import Ridge Regression and related tools
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Encode the TYPE column into numeric labels
label_encoder = LabelEncoder()
training_set['TYPE_NUM'] = label_encoder.fit_transform(training_set['TYPE'])
# Encode test labels using the same encoder for consistency
test_features = testing_set.drop(columns=['TYPE']).values
test_true_labels = label_encoder.transform(testing_set['TYPE'])

ridge_model = Ridge(alpha=1)  # alpha is the regularization strength
ridge_model.fit(test_features,test_true_labels)
# Predict using the Ridge regression model
predicted_numeric_labels = np.round(ridge_model.predict(test_features)).astype(int)

# Ensure predicted labels are consistent with the training label encoding
predicted_numeric_labels = np.clip(predicted_numeric_labels, 0, len(label_encoder.classes_) - 1)
predicted_labels = label_encoder.inverse_transform(predicted_numeric_labels)

ridge_accuracy = accuracy_score(testing_set['TYPE'], predicted_labels)
ridge_accuracy

# Step 1: Calculate centroids for each diagnosis type
diagnoses = training_set['TYPE'].unique()
centroids = []
for diagnosis in diagnoses:
    # Extract features for the current diagnosis
    diagnosis_data = training_set[training_set['TYPE'] == diagnosis].drop(columns=['TYPE']).values
    # Compute the centroid (mean of features for the diagnosis)
    centroids.append(np.mean(diagnosis_data, axis=0))

# Convert centroids to a dictionary for easier mapping
diagnosis_to_centroid = {diagnosis: centroid for diagnosis, centroid in zip(diagnoses, centroids)}

# Step 2: Assign test data points to the nearest centroid and predict labels
test_features = testing_set.drop(columns=['TYPE']).values
true_labels = testing_set['TYPE']
predicted_labels = []

for point in test_features:
    # Compute distances to all centroids
    distances = {diagnosis: np.linalg.norm(point - centroid) for diagnosis, centroid in diagnosis_to_centroid.items()}
    # Assign the diagnosis with the nearest centroid
    predicted_labels.append(min(distances, key=distances.get))
# Step 3: Evaluate overall accuracy
overall_accuracy = accuracy_score(true_labels, predicted_labels)


overall_accuracy

# Let's Test Accuracy 
from sklearn.metrics import accuracy_score
test_features = testing_set.drop(columns=['TYPE']).values
true_labels = testing_set['TYPE']

# Map each diagnosis to its centroid from the training phase
diagnosis_to_centroid = {diagnosis: centroid[0] for diagnosis, centroid in zip(diagnoses, centroids)}

# Predict labels for the test set
predicted_labels = []

for point in test_features:
    # Calculate distances to all centroids
    distances = {diagnosis: np.linalg.norm(point - centroid) for diagnosis, centroid in diagnosis_to_centroid.items()}
    # Assign the diagnosis with the closest centroid
    predicted_labels.append(min(distances, key=distances.get))

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

accuracy
# Recompute centroids for training data
centroids = []
for diagnosis in diagnoses:
    diagnosis_data = training_set[training_set['TYPE'] == diagnosis].drop(columns=['TYPE']).values
    kmean = sklearn.cluster.KMeans(n_clusters=1, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=42, algorithm='lloyd')
    # Calculating kmean separately
    kmean.fit(diagnosis_data)
    centroids.append(kmean.cluster_centers_)

# Map each diagnosis to its centroid
diagnosis_to_centroid = {diagnosis: centroid[0] for diagnosis, centroid in zip(diagnoses, centroids)}

# Calculate accuracy per diagnosis category in the test set
category_accuracies = {}
for diagnosis in diagnoses:
    # First filter rows for the current diagnosis
    category_data = testing_set[testing_set['TYPE'] == diagnosis]
    
    # Prepare features and true labels for this category
    category_features = category_data.drop(columns=['TYPE']).values
    category_labels = category_data['TYPE']
    
    # Predict labels for this category
    category_predictions = []
    for point in category_features:
        distances = {diag: np.linalg.norm(point - centroid) for diag, centroid in diagnosis_to_centroid.items()}
        category_predictions.append(min(distances, key=distances.get))
    
    # Calculate accuracy for this category
    category_accuracy = accuracy_score(category_labels, category_predictions)
    category_accuracies[diagnosis] = category_accuracy

# Display accuracies by category
category_accuracies_df = pd.DataFrame(category_accuracies.items(), columns=["Diagnosis", "Accuracy"])