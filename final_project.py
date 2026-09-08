'''
from keras.datasets import boston_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

'''
#(i)
''' 
ridge_reg_model = Ridge(alpha = 1.0)
ridge_reg_model.fit(x_train, y_train)

y_test_pred = ridge_reg_model.predict(x_test)
m_sq_e = mean_squared_error(y_test, y_test_pred)

print("Mean Squared Error:", m_sq_e)

'''
#(ii)
'''
kmeans = KMeans(n_clusters = 3)
training_clusters = kmeans.fit_predict(x_train)
training_centroids = kmeans.cluster_centers_

plt.figure()
colors = ["red", "blue", "green"]
for cluster in range(3):
    cluster_points = x_train[training_clusters == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label = f"Cluster {cluster + 1}", color = colors[cluster])
plt.title("K-Means CLustering on the Training Data")
plt.legend()
plt.show()

'''
#(iii)
'''

testing_clusters = kmeans.predict(x_test)
models = {}
cluster_m_sq_e = []

for cluster in range(3):
    cluster_i = (training_clusters == cluster)
    x_training_cluster = x_train[cluster_i]
    y_training_cluster = y_train[cluster_i]
    
    ridge_reg = Ridge(alpha = 1.0)
    ridge_reg.fit(x_training_cluster, y_training_cluster)
    models[cluster] = ridge_reg

    test_cluster_i = (testing_clusters == cluster)
    x_testing_cluster = x_test[test_cluster_i]
    y_testing_cluster = y_test[test_cluster_i]
    
    y_testing_pred = ridge_reg.predict(x_testing_cluster)
    
    c_m_sq_e = mean_squared_error(y_testing_cluster, y_testing_pred)
    cluster_m_sq_e.append(c_m_sq_e)
    
total_m_sq_e = np.mean(cluster_m_sq_e)
print("Total Mean Squared Error:", total_m_sq_e)
'''
from keras.datasets import boston_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
ridge_model = Ridge(alpha=1.0)  
ridge_model.fit(x_train,y_train)
y_pred = ridge_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Ridge Regression): ", mse)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(x_train)
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
for cluster in np.unique(clusters):
    cluster_points = x_train[clusters == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster+1}")
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.xlabel("Feature 1 (X1)")
plt.ylabel("Feature 2 (X2)")
plt.title("K-Means Clustering (k=3) on Training Data")
plt.legend()
plt.show()


test_clusters = kmeans.predict(x_test)
models = {}
cluster_mse = []
for cluster in range(3):
    cluster_indices = (clusters == cluster)
    x_cluster_train = x_train[cluster_indices]
    y_cluster_train = y_train[cluster_indices]
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_cluster_train, y_cluster_train)
    models[cluster] = ridge_model
    
    test_cluster_indices = (test_clusters == cluster)
    x_cluster_test = x_test[test_cluster_indices]
    y_cluster_test = y_test[test_cluster_indices]
    y_pred = ridge_model.predict(x_cluster_test)
    
    mse = mean_squared_error(y_cluster_test, y_pred)
    cluster_mse.append(mse)

total_mse = np.mean(cluster_mse)
print("Total Mean Squared Error: ", total_mse)



