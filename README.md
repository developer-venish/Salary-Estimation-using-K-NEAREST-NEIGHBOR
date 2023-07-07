# Salary-Estimation-using-K-NEAREST-NEIGHBOR
ML Python Project

---------------------------------------------------------------------------------------
Plots the error rate versus k to visualize the optimal value of k with the lowest error
![](https://github.com/developer-venish/Salary-Estimation-using-K-NEAREST-NEIGHBOR/blob/main/matplot.png)
---------------------------------------------------------------------------------------

# Preview
![](https://github.com/developer-venish/Salary-Estimation-using-K-NEAREST-NEIGHBOR/blob/main/demo.png)
---------------------------------------------------------------------------------------

# Accuracy of the model
![](https://github.com/developer-venish/Salary-Estimation-using-K-NEAREST-NEIGHBOR/blob/main/accuracy.png)
---------------------------------------------------------------------------------------

All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!
---------------------------------------------------------------------------------------


# Working of The Code

This code snippet performs the following tasks:

1. Installs the `pandas` library and imports necessary packages:
   - `pandas` for data manipulation and analysis.
   - `numpy` for numerical operations.
   - `files` from `google.colab` for uploading files in Google Colab environment.

2. Uploads a CSV file (`salary1.csv`) using the `files.upload()` method.

3. Reads the uploaded CSV file into a pandas DataFrame called `dataset`.

4. Prints the shape of the dataset using `dataset.shape` and the first 5 rows of the dataset using `dataset.head(5)`.

5. Converts the values in the "income" column from categorical labels ("<=50K" and ">50K") to numeric values (0 and 1) using `dataset['income'].map()` method.

6. Prints the first 5 rows of the dataset after the "income" column conversion.

7. Extracts the feature matrix (`X`) by selecting all columns except the last one, and the target array (`Y`) by selecting only the last column from the dataset.

8. Installs the `scikit-learn` library for machine learning.

9. Splits the dataset into training and testing sets using `train_test_split()` from `sklearn.model_selection`. The training set consists of 75% of the data, and the testing set consists of the remaining 25%.

10. Applies feature scaling on the training and testing sets using `StandardScaler` from `sklearn.preprocessing`.

11. Performs model training and evaluation using the k-nearest neighbors (KNN) algorithm:
    - Initializes an empty list `error` to store the error rates.
    - Iterates over different values of `k` from 1 to 40.
    - Trains a KNN classifier with `n_neighbors=i`.
    - Predicts the labels for the testing set.
    - Calculates the mean error and appends it to the `error` list.
    - Plots the error rate versus `k` to visualize the optimal value of `k` with the lowest error.

12. Trains a KNN classifier with `n_neighbors=5`, `metric='minkowski'`, and `p=2` using the preprocessed training set.

13. Takes input for a new employee's age, education, capital gain, and hours per week.

14. Creates a new employee data (`newEmp`) and predicts the salary category for the new employee using the trained model.

15. Predicts the labels for the testing set (`Y_pred`).

16. Computes the confusion matrix and accuracy score of the model using `confusion_matrix()` and `accuracy_score()` from `sklearn.metrics`.

17. Prints the confusion matrix and the accuracy of the model.

---------------------------------------------------------------------------------------
K-Nearest Neighbors (KNN) is a non-parametric and supervised machine learning algorithm used for classification and regression tasks. It is a simple and intuitive algorithm that classifies new data points based on the majority class of its k nearest neighbors in the feature space.

The algorithm works as follows:

1. Select a value for k, which represents the number of nearest neighbors to consider.
2. For a given test data point, calculate the distances to all training data points.
3. Identify the k nearest neighbors based on the shortest distances.
4. For classification, assign the majority class label among the k nearest neighbors to the test data point. For regression, take the average of the target values of the k nearest neighbors.
5. Output the predicted class label or regression value for the test data point.

KNN is a lazy learning algorithm, meaning that it does not explicitly build a model during the training phase. Instead, it stores all training instances and performs computation at the time of prediction. KNN is often used for its simplicity and ability to handle non-linear decision boundaries. However, it can be sensitive to the choice of k and the distance metric used.
