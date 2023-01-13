from functions import get_data, logRegrPredict_A1,img_SVM_A2, B1_feature_label, B2_feature, B2_label, MLP
import os
from sklearn.preprocessing import StandardScaler
tr_X, tr_Y_gender, tr_Y_smile, te_X, te_Y_gender, te_Y_smile = get_data()

# reshape data for A1 and run A1
X_train, y_train, X_test, y_test = tr_X.reshape((4795, 68*2)), tr_Y_gender, te_X.reshape((969, 68*2)), te_Y_gender
Test_Accuracy = logRegrPredict_A1(X_train, y_train, X_test, y_test)
# # reshape data for A2 and run A2
X_train, y_train, X_test, y_test = tr_X.reshape((4795, 68*2)), tr_Y_smile, te_X.reshape((969, 68*2)), te_Y_smile
Test_Accuracy = img_SVM_A2(X_train, y_train, X_test, y_test)
# using path to read data for B1
basedir_tr = './Dataset/cartoon_set'
images_dir_tr = os.path.join(basedir_tr,'img')
labels_filename_tr = 'labels.csv'
X_train, y_train_shape = B1_feature_label(images_dir=images_dir_tr, labels_filename=labels_filename_tr, basedir=basedir_tr)
X_train = X_train.reshape(8194,68*2)

basedir_te = './Dataset/cartoon_set_test'
images_dir_te = os.path.join(basedir_te,'img')
labels_filename_te = 'labels.csv'
X_test, y_test_shape = B1_feature_label(images_dir=images_dir_te, labels_filename=labels_filename_te, basedir=basedir_te)
X_test = X_test.reshape(2041,68*2)
# Test model for B1
normlisor = StandardScaler()
X_train = normlisor.fit_transform(X_train)
X_test = normlisor.fit_transform(X_test)
y_pred = MLP(X_train=X_train,y_train=y_train_shape,X_test=X_test,y_test=y_test_shape)
# Get data for B2
X_train = B2_feature(len = 10000, path = './Dataset/cartoon_set/img/')
X_test = B2_feature(len = 2500, path = './Dataset/cartoon_set_test/img/')
y_train_color = B2_label(path = './Dataset/cartoon_set/labels.csv')
y_test_color = B2_label(path = './Dataset/cartoon_set_test/labels.csv')
normlisor = StandardScaler()
# Test model for B2
X_train = normlisor.fit_transform(X_train)
X_test = normlisor.fit_transform(X_test)
y_pred = MLP(X_train=X_train,y_train=y_train_color,X_test=X_test,y_test=y_test_color)