import seaborn as sn
import numpy as np
import cv2 as cv2
import tqdm as tqdm
import os
from glob import glob

from PIL import Image
from torch import nn
import torch as t
import torchvision.models as models
from torchvision import transforms
from torchsummary import summary
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier


root_path = 'FILE_PATH'


# split into subfolders based on class label
subfolders = sorted(glob(root_path + '/*'))
label_names = [p.split('/')[-1] for p in subfolders]
print(label_names)

# Load in the pretrained resnet101 NN
resnet101 = models.resnet101(pretrained=True)

# Slices the model
def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])

model_conv_features = slice_model(resnet101, to_layer=-1).to('cpu')
summary(model_conv_features, input_size=(3, 224, 224))


# Preprocesses the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Make sure images are of the correct data type
def retype_image(in_img):
  if np.max(in_img) > 1:
    in_img = in_img.astype(np.uint8)
  else:
    in_img = (in_img * 255.0).astype(np.uint8)
  return in_img

# Put the model in evaluation mode
resnet101.eval()


images = []
labels = []
flipped_images = []
flipped_labels = []

# Load the images and stores them in a array, same with the labels
for i in range(len(subfolders)):
    img_paths = sorted(glob(os.path.join(subfolders[i], '*.jpg')))
    img_paths = [img_path for img_path in img_paths if os.path.isfile(img_path)]  # Filter out directories
    count = 0
    if img_paths:
        count = (len(img_paths)//3)*2
        for img_path in img_paths:
            if count > 1:
                img = cv2.imread(img_path)
                images.append(img)
                labels.append(i)
                count -= 1
                
# Load the images but flipped (incresing the data sample size)
flipped_images = []
for i in range(len(subfolders)):
    img_paths = sorted(glob(os.path.join(subfolders[i], '*.jpg')))
    img_paths = [img_path for img_path in img_paths if os.path.isfile(img_path)]  # Filter out directories
    count = 0
    if img_paths:
        count = (len(img_paths)//3)*2
        for img_path in img_paths:
            if count > 1:
                img = cv2.imread(img_path)
                img = cv2.flip(img, 1)
                flipped_images.append(img)
                flipped_labels.append(i)
                count -= 1
              
# Obtain the features of both sets of images using resnet101  
features = []
flipped_features = []
i = 0
labels = np.array(labels)
count = 0
start = time.time()
for image in tqdm.tqdm(images, desc='Loading features'):
  proc_img = preprocess(Image.fromarray(image))
  feat = model_conv_features(proc_img.unsqueeze(0).to('cpu')).squeeze().detach().numpy()
  features.append(feat)
for image in tqdm.tqdm(flipped_images, desc='Loading features'):
  proc_img = preprocess(Image.fromarray(image))
  feat = model_conv_features(proc_img.unsqueeze(0).to('cpu')).squeeze().detach().numpy()
  flipped_features.append(feat)

nn_features = np.array(features)
nn_flipped_features = np.array(flipped_features)
labels = np.array(labels)



# Run features through PCA
print(nn_features.shape)
pca = PCA(n_components=100)

flipped_pca = PCA(n_components=100)

new_feature = pca.fit_transform(nn_features)
new_flip_features = pca.fit_transform(nn_flipped_features)
end = time.time()
print('Time to create feature vectors:', end - start)
# Split the train/test data from both sets of features
X_train, X_test, y_train, y_test = train_test_split(
    new_feature,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42,
    )
X_flip_train, X_flip_test, y_flip_train, y_flip_test = train_test_split(
    new_flip_features,
    flipped_labels,
    test_size=0.2,
    stratify=flipped_labels,
    random_state=42
    )

# Combine all the train and test sets together
X_combined_train = np.concatenate((X_train, X_flip_train), axis=0)
X_combined_test = np.concatenate((X_test, X_flip_test), axis=0)
y_combined_train = np.concatenate((y_train, y_flip_train), axis=0)
y_combined_test = np.concatenate((y_test, y_flip_test), axis=0)


# First classifer, this one does not print out. It was 90% accurate as well but it was too
# prone to overfitting
# bst = XGBClassifier(n_estimators=150,
#                     max_depth=2, 
#                     learning_rate=.12, 
#                     objective='multi:softprob', 
#                     num_class=6, 
#                     random_state=42)
# bst.fit(X_combined_train, y_combined_train)
# y_pred = bst.predict(X_combined_test)


#* The classifier for fitting the data.

print('My best classifier: the standard SVC: ')
start = time.time()
clf = make_pipeline(StandardScaler(), SVC(gamma='scale', C=1, kernel='rbf'))
clf.fit(X_combined_train, y_combined_train)

# Report accuracy for each class
y_pred = clf.predict(X_combined_test)
accuracy = accuracy_score(y_combined_test, y_pred)


# Plot the results as a confusion matrix
C = confusion_matrix(y_combined_test, y_pred)
sn.heatmap(C, annot=True, cmap='Greens', fmt='0.0f')
report = classification_report(y_combined_test, y_pred)
end = time.time()
print('Time to fit and test data: ', end - start)

# A more detailed report on how my classifier did per class
print(report)

# Testing the training data
start = time.time()
y_pred = clf.predict(X_combined_train)
report = classification_report(y_combined_train, y_pred)
end = time.time()

# Generate the confusion matrix
C = confusion_matrix(y_combined_train, y_pred)
sn.heatmap(C, annot=True, cmap='Reds', fmt='0.0f')
print('Time to test the training data: ', end - start)

print(report)


param_grid = {
    'svc__C': [0.1, 1, 3, 7, 10],  #
    'svc__gamma': ['scale', 'auto'], 
    'svc__kernel': ['linear', 'rbf', 'poly']  
}

# Perform grid search for the SVM, commented out as it has already been ran
# The best parameters were: {'svc__C': 1, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}

#? grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=1)

# # Fit grid search to the combined training data
#? grid_search.fit(X_combined_train, y_combined_train)

# # Print the best parameters found
#? print("Best Parameters:", grid_search.best_params_)

# # Get the best estimator
#? best_clf = grid_search.best_estimator_

# # Evaluate the best classifier on the combined test set
#? accuracy = best_clf.score(X_combined_test, y_combined_test)
#? print("Accuracy:", accuracy)
