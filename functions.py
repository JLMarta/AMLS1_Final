# function 1: Extract features from Images
import os
import numpy as np
from keras_preprocessing import image
import cv2
import dlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
# PATH TO ALL IMAGES

global basedir, image_paths, target_size
basedir_tr = './Dataset/dataset_AMLS_22-23/celeba'
images_dir_tr = os.path.join(basedir_tr,'img')
labels_filename_tr = 'labels.csv'
basedir_te = './Dataset/dataset_AMLS_22-23_test/celeba_test'
images_dir_te = os.path.join(basedir_te,'img')
labels_filename_te = 'labels.csv'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(images_dir, labels_filename, basedir):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    path = os.listdir(images_dir)
    path.sort(key=lambda x:int(x[:-4]))
    image_paths = [os.path.join(images_dir, l) for l in path]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r') 
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    smile_labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_gender_labels = []
        all_smile_labels = []
        
        for img_path in image_paths:
            file_name= img_path.split('.')[-2].split('\\')[-1]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_gender_labels.append(gender_labels[file_name])
                all_smile_labels.append(smile_labels[file_name])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_gender_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    smile_labels = (np.array(all_smile_labels) + 1)/2 # simply converts the -1 into 0, so no smile=0 and smile=1
    return landmark_features, gender_labels, smile_labels


def get_data():
    
    X_tr, y_gender_tr, y_smile_tr = extract_features_labels(images_dir_tr,labels_filename_tr,basedir_tr)
    X_te, y_gender_te, y_smile_te = extract_features_labels(images_dir_te,labels_filename_te,basedir_te)
    
    Y_gender_tr = np.array(y_gender_tr).T 
    Y_smile_tr = np.array(y_smile_tr).T

    tr_X = X_tr
    tr_Y_gender = Y_gender_tr
    tr_Y_smile = Y_smile_tr

    Y_gender_te = np.array(y_gender_te).T
    Y_smile_te = np.array(y_smile_te).T

    te_X = X_te
    te_Y_gender = Y_gender_te
    te_Y_smile = Y_smile_te
    
    return tr_X, tr_Y_gender, tr_Y_smile, te_X, te_Y_gender, te_Y_smile

def B1_feature_label(images_dir, labels_filename, basedir):
    path = os.listdir(images_dir)
    path.sort(key=lambda x:int(x[:-4]))
    image_paths = [os.path.join(images_dir, l) for l in path]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r') 
    lines = labels_file.readlines()
    shape_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_shape_labels = []
        for img_path in image_paths:
            file_name= img_path.split('.')[-2].split('\\')[-1]
            # load image
            img = image.img_to_array(
                image.load_img(img_path, target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_shape_labels.append(shape_labels[file_name])
    img_features = np.array(all_features)
    shape_labels = np.array(all_shape_labels)
    return img_features,shape_labels


def B2_feature(len,path):
    dataset_img = []
    for img_name in range(len):
    # defining the image path
        image_path = path + str(img_name) + '.png'
    # reading the image
        img = cv2.imread(image_path)
    # normalizing the pixel values
        img = img[240:290,150:350,:]
    # converting the type of pixel to float 32
        img = img.astype('float32')
    # appending the image into the list
        dataset_img.append(img)
    dataset_img = np.array(dataset_img)
    dataset_img = dataset_img.reshape(len,50*200*3)
    return dataset_img

def B2_label(path):
    color_label = pd.read_csv(path,delimiter = '\t')
    color_label = color_label['eye_color']
    color_label = color_label.to_numpy()
    return color_label

def Hyper_Paramter_Tune(model, grid, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_params_


def MLP(X_train, y_train, X_test, y_test):

    mlp = MLPClassifier(solver = 'adam', activation = 'relu', alpha = 0.1, hidden_layer_sizes = (200,200),
                        verbose = True, random_state = 1, max_iter = 200, learning_rate='constant',
                        batch_size = 100, learning_rate_init = 0.01)
    mlp.fit(X_train,y_train)
    y_pred = mlp.predict(X_test)
    a = mlp.loss_curve_
    plt.xlabel('Number of Interations')
    plt.ylabel('Train loss')
    plt.title('Train loss vs iteration')
    plt.plot(a)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    return y_pred

# Log Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
def logRegrPredict_A1(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(solver='liblinear', max_iter = 400, C = 0.1)
    logreg.fit(X_train, y_train)
    y_pred= logreg.predict(X_test)
    Test_Accuracy = accuracy_score(y_test,y_pred)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    return Test_Accuracy

# KNN
from sklearn.neighbors import KNeighborsClassifier
def KNNClassifier_A2(X_train, y_train, X_test, y_test):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=29, weights='distance', metric='manhattan')
    neigh.fit(X_train, y_train) # Fit KNN model


    y_pred = neigh.predict(X_test)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    return y_pred

# Best A2 SVM 
from sklearn import svm
def img_SVM_A2(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='rbf', C=50, gamma='scale')

    classifier.fit(training_images, training_labels)

    y_pred = classifier.predict(test_images)

    Test_Accuracy = accuracy_score(test_labels,y_pred)
    print('Accuracy on test set: '+str(accuracy_score(test_labels,y_pred)))
    return Test_Accuracy

from sklearn import tree
def DecisionTree_A2(X_train, y_train, X_test, y_test):
    tree_params={'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 20, 'min_samples_split': 16}
    Tree = tree.DecisionTreeClassifier( **tree_params )
    Tree.fit(X_train, y_train)
    y_pred =  Tree.predict(X_test)
    # print(f'Test feature {X_test}\n True class {y_test[0]}\n predict class {y_pred[0]}')
    print(confusion_matrix(y_test, y_pred))
    print('Accuracy on train set: '+str(accuracy_score(y_train,Tree.predict(X_train))))
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))#text report showing the main classification metrics
    return  y_pred

from sklearn.ensemble import RandomForestClassifier

def RandomForest_A1(X_train, y_train, X_test, y_test):
    Forest=RandomForestClassifier(n_estimators=1000, max_features='log2')
    Forest.fit(X_train, y_train)
    y_pred = Forest.predict(X_test)
    print('Random Forest test Accuracy: '+str(accuracy_score(y_test,y_pred)))
    return y_pred

from sklearn.ensemble import BaggingClassifier
def baggingClassifierML_A2(X_train, y_train, X_test, y_test):

    #Create KNN object with a K coefficient
    bagmodel=BaggingClassifier(n_estimators=1000,max_samples=0.5, max_features=4,random_state=1)
    bagmodel.fit(X_train, y_train) # Fit KNN model
    y_pred = bagmodel.predict(X_test)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    return y_pred

from sklearn.ensemble import AdaBoostClassifier
def boostingClassifierML_A1(X_train, y_train, X_test, y_test):
    # AdaBoost takes Decision Tree as its base-estimator model by default.
    boostmodel=AdaBoostClassifier(n_estimators=1000)
    boostmodel.fit(X_train , y_train,sample_weight=None) # Fit KNN model
    y_pred = boostmodel.predict(X_test)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    return y_pred

from sklearn.neural_network import MLPClassifier
def MLP_A1(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(solver = 'adam', activation = 'relu', alpha = 1e-4, hidden_layer_sizes = (150,150,150),
                        random_state = 1,max_iter = 300, learning_rate='constant')
    mlp.fit(X_train,y_train)
    y_pred = mlp.predict(X_test)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
    print(classification_report(y_test,y_pred))#text report showing the main classification metrics
    return y_pred