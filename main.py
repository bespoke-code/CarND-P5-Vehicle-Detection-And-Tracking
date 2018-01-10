import cv2
import numpy as np
from sklearn import svm, model_selection, utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage.measurements import label
import data_manipulation
import time
from heatmap_utils import HeatmapHistory
from moviepy.editor import VideoFileClip
from glob import glob
from matplotlib import pyplot as plt

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


# Define components required for further work
heatmapHistory = HeatmapHistory(threshold=11)
X_scaler = StandardScaler()
pca = PCA(n_components=300, whiten=True)
svc = svm.SVC(kernel='rbf')

# Define feature extraction parameters
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [380, 720]  # Min and max in y to search in slide_window()


def process_frame(frame):
    # process video frame by frame
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    frame = frame.astype(np.float32)/255.

    xy_windows = [(64, 64), (96, 96), (128, 128)]
    y_regions = [[380, 520], [400, 550], [470, 600]]
    #colors = [(1.,0,0), (0,1.,0), (0,0,1.)]

    #draw_image = np.copy(frame)

    # TODO: Implement this to save time in the future?
    # hog_image = data_manipulation.get_hog_features(draw_image, orient, pix_per_cell, cell_per_block,
    #                                                vis=False, feature_vec=True)

    hot_windows_for_frame = []
    for j in range(len(xy_windows)):
        windows = data_manipulation.slide_window(frame, x_start_stop=[80, 1280], y_start_stop=y_regions[j],
                                                 xy_window=xy_windows[j], xy_overlap=(0.5, 0.5))

        # use trained classifier, model fitter etc.
        hot_windows = search_windows(frame, windows, svc, X_scaler, pca, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)
        #print(len(hot_windows), 'windows found!')
        hot_windows_for_frame.extend(hot_windows)

        #draw_image = data_manipulation.draw_boxes(draw_image, hot_windows, color=colors[j], thick=j + 2)
        #plt.imshow(draw_image)
    #plt.show()

    # prepare heatmap
    heatmapHistory.add(hot_windows_for_frame)
    #heatmapHistory.visualize_heatmap()

    time_series_heatmap = heatmapHistory.get_heatmap()
    # Label each part of the heatmap
    labeled_windows = label(time_series_heatmap)
    # Draw a box around each labeled part!
    frame = frame * 255.
    frame = np.clip(frame, 0, 255)
    drawn_frame = data_manipulation.draw_labeled_bboxes(frame, labeled_windows)
    return drawn_frame


if __name__ == '__main__':
    # Get each image's path
    #cars = data_manipulation.load_data_jpg('./small_dataset/vehicles_smallset/cars*/')
    cars = data_manipulation.load_data_png('./dataset/vehicles/*/')

    #notcars = data_manipulation.load_data_jpg('./small_dataset/non-vehicles_smallset/notcars*/')
    notcars = data_manipulation.load_data_png('./dataset/non-vehicles/*/')

    # I really like a balanced dataset.
    datapoints_count = min([len(cars), len(notcars)])
    #print('Using', 2*datapoints_count, 'data points for training (balanced).')

    cars = utils.shuffle(cars)[:datapoints_count]
    notcars = utils.shuffle(notcars)[:datapoints_count]

    # Feature extraction
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

    #  Visualize the feature vector. Used to debug the output to see if all goes according to plan.
    #data_manipulation.visualize(X[0])

    # print('Performing scaling...')
    # Scaling the features
    X_scaler.fit(X)
    scaled_X = X_scaler.transform(X)
    #data_manipulation.visualize(scaled_X[0])
    # Informative, print the mean and variance vectors for each scaled feature
    #print('Mean, variance:', X_scaler.mean_, X_scaler.var_)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # An idea from Sebastian Raschka's Python Machine Learning, 1st Ed. book:
    # Compressing Data via Dimensionality Reduction, Ch. 5 - LDA and PCA!
    # Use LDA or PCA to select meaningful features and shrink the feature vector.
    pca.fit(scaled_X)
    print('Component analysis done.')

    scaled_pca_X = pca.transform(scaled_X)
    #print('Scaled pca feature len:', scaled_pca_X.shape)
    #print('Scaled pca mean:', np.mean(scaled_pca_X))
    #data_manipulation.visualize(scaled_pca_X[0])

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        scaled_pca_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # TODO: Use a Grid search (?) to tune params to the max.
    # Using rbf SVC kernel
    # Check the training time for the SVC
    t = time.time()

    svc.fit(X_train, y_train)
    print(svc.n_support_)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the accuracy of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # See results on test images (for debugging)
    #test_images = sorted(glob('examples/frame*.jpg'))
    #
    #for img in test_images:
    #    img_result = plt.imread(img)
    #    img_result = process_frame(img_result)
    #    plt.imshow(img_result)
    #    plt.show()


    out_dir='./videos/'
    output = out_dir+'generated_project_video.mp4'

    clip = VideoFileClip("../CarND-P4-Advanced-Lane-Finding/project_video.mp4") #.subclip(35,43)
    out_clip = clip.fl_image(process_frame)
    # Add frame back to video and save it
    out_clip.write_videofile(output, audio=False)

    # Final output (terminal)
    # Component analysis done.
    # Using: 9 orientations 16 pixels per cell and 2 cells per block
    # Feature vector length: 300
    # [2228 2703]
    # 47.48 Seconds to train SVC...
    # Test Accuracy of SVC =  0.9915