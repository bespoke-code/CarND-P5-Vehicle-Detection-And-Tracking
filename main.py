from skimage import feature
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from sklearn import svm, model_selection, utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import data_manipulation
import time
from glob import glob
from heatmap_utils import HeatmapHistory


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = data_manipulation.bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = data_manipulation.color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(data_manipulation.get_hog_features(feature_image[:, :, channel],
                                                                       orient, pix_per_cell, cell_per_block,
                                                                       vis=False, feature_vec=True))
        else:
            hog_features = data_manipulation.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                              pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, comp_analyzer, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        test_features = comp_analyzer.transform(test_features)
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


heatmapHistory = HeatmapHistory(threshold=5)

def process_frame(frame):
    global heatmapHistory


    # load data
    # get feature set
    # get trained classifier
    # prepare heatmap
    # process video frame by frame


if __name__ == '__main__':
    heatmapHistory = HeatmapHistory(threshold=1)
    # Get the dataset in place
    #cars = data_manipulation.load_data_jpg('./small_dataset/vehicles_smallset/cars1/')
    cars = data_manipulation.load_data_png('./dataset/vehicles/*/')
    # y_cars1 = [[1] for i in range(len(X_cars1))]
    #notcars = data_manipulation.load_data_jpg('./small_dataset/non-vehicles_smallset/notcars1/')
    notcars = data_manipulation.load_data_png('./dataset/non-vehicles/*/')
    # y_not_cars1 = [[0] for i in range(len(X_not_cars1))]

    # Check dataset state:
    # - balanced
    datapoints_count = min([len(cars), len(notcars)])
    print('Using', 2*datapoints_count, 'data points for training (balanced).')
    cars = utils.shuffle(cars)[:datapoints_count]
    # y_cars1 = y_cars1[:datapoints_count]
    notcars = utils.shuffle(notcars)[:datapoints_count]
    # y_not_cars1 = y_not_cars1[:datapoints_count]
    # - colorspace of each image
    # - size of each image
    # Feature extraction from the dataset

    color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 12  # HOG orientations
    pix_per_cell = 16  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [400, 720]  # Min and max in y to search in slide_window()

    car_features = data_manipulation.extract_features(cars, color_space=color_space,
                                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                                      orient=orient, pix_per_cell=pix_per_cell,
                                                      cell_per_block=cell_per_block,
                                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                      hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = data_manipulation.extract_features(notcars, color_space=color_space,
                                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                                         orient=orient, pix_per_cell=pix_per_cell,
                                                         cell_per_block=cell_per_block,
                                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                         hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    print('Performing scaling...')
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print('Mean, variance:', X_scaler.mean_, X_scaler.var_)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # An idea from Sebastian Raschka's Python Machine Learning, 1st Ed. book:
    # Compressing Data via Dimensionality Reduction, Ch. 5 - LDA and PCA!
    # Use LDA or PCA to select meaningful features and shrink the feature vector.
    pca = PCA(n_components=300, whiten=True)
    pca.fit(scaled_X)
    print('Component analysis done.')

    scaled_pca_X = pca.transform(scaled_X)
    print('Scaled pca feature len:', scaled_pca_X.shape)
    print('Scaled pca mean:', np.mean(scaled_pca_X))
    data_manipulation.visualize(scaled_pca_X[0])

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        scaled_pca_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # TODO: Use a Grid search (?) to tune params to the max.
    # Use a rbf SVC
    svc = svm.SVC(kernel='rbf')
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    test_images = sorted(glob('examples/frame*.jpg'))
    print(test_images)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255
    xy_windows = [(64, 64), (96, 96), (128, 128), (256, 256)]
    y_regions = [[400, 520], [400, 550], [470, 600], [400, 720]]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (127, 127, 255)]
    for i in range(len(test_images)):
        image = mpimg.imread(test_images[i])
        print('loaded image ', test_images[i])
        draw_image = np.copy(image)
        # hog_image = data_manipulation.get_hog_features(draw_image, orient, pix_per_cell, cell_per_block,
        #                                                vis=False, feature_vec=True)
        start_time = time.time()
        for j in range(len(xy_windows)):
            windows = data_manipulation.slide_window(image, x_start_stop=[None, None], y_start_stop=y_regions[j],
                                                     xy_window=xy_windows[j], xy_overlap=(0.5, 0.5))

            hot_windows = search_windows(image, windows, svc, X_scaler, pca, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)
            print(len(hot_windows), 'windows found!')
            draw_image = data_manipulation.draw_boxes(draw_image, hot_windows, color=colors[j], thick=j + 2)
            #heatmapHistory.add(hot_windows)
            #heatmapHistory.visualize_heatmap()
            plt.imshow(draw_image)
        print('Time to get all window candidates:', time.time() - start_time)
        plt.show()