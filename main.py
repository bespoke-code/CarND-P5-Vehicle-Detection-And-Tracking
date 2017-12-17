from skimage import feature
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from sklearn import svm, model_selection
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    # Check dataset state:
    # - balanced
    # - colorspace of each image
    # - size of each image
    # Feature extraction from the dataset



    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Do a train-test split
    #X_train = model_selection.train_test_split()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(scaled_X, y, test_size=0.3, random_state=np.random.randint(0, 100))

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()
    clf = model_selection.GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)

    print(clf.best_params_, clf.best_estimator_, clf.best_score_)

    print('Test Accuracy of SVC = ', clf.score(X_test, y_test))