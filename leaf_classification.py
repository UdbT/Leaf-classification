import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg       # reading images to numpy arrays
import random
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
    
collection = pd.read_csv('output_train.csv')

outter_train = pd.read_csv('train.csv')
test = pd.read_csv('output_test.csv')

# For null checking
sel_df = collection
sel_df = sel_df.drop('species',1)

def rescaling(column):
    collection[column] = (collection[column]-collection[column].min())/(collection[column].max() - collection[column].min())
def rescaling_test(column):
    test[column] = (test[column]-test[column].min())/(test[column].max() - test[column].min())
    
rescaling_test('Mean')
rescaling_test('Variance')
rescaling_test('total_maxima')
rescaling_test('total_minima')
rescaling_test('axis-y/axis-x')
rescaling_test('area/rounded_length')

rescaling('Mean')
rescaling('Variance')
rescaling('total_maxima')
rescaling('total_minima')
rescaling('axis-y/axis-x')
rescaling('area/rounded_length')
    
def encode(train,test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id 
    
    train = train.loc[:,'Mean':'area/rounded_length']
    
    return train, labels, classes, test, test_ids, le

train, labels, classes, test, test_ids, le = encode(collection, test)

# Check Null
plt.figure(1,figsize=(15,9)) 
ax = sns.heatmap(sel_df.isnull(),yticklabels=False,cbar=False,cmap='viridis') 
ax.set_xticklabels(sel_df,rotation =90) 
ax.figure.tight_layout()

#Find most accurate clssifier
classifiers = [
    KNeighborsClassifier(3),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

# 10 fold cross validation
kf = KFold(n_splits=10)
X = np.array(train)
y = np.array(labels)
collect = pd.DataFrame(columns=['Accuracy'])
for clf in classifiers:

    list_accuracy = []
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        
        #Accuracy
        accuracy = accuracy_score(y_test, predicted)
        list_accuracy.append(accuracy)
    
    collect = collect.append(pd.DataFrame([[np.array(list_accuracy).mean()*100]],index=[clf.__class__.__name__],columns=['Accuracy']))
    print("Average accuracy = %f\n" %(np.array(list_accuracy).mean()*100))

collect = collect.sort_values('Accuracy',ascending=False)

# Find the most accurate classifier
for clf in classifiers:
    if clf.__class__.__name__ == collect.idxmax()[0]:
        selected_clf = clf
        break

#selected_clf = LinearDiscriminantAnalysis()
# Fit data to classification madel
selected_clf.fit(X,y)

# Result
predicted = selected_clf.predict(test.loc[:,'Mean':'area/rounded_length'])

# Map image id with predicted result
predicted_table = pd.DataFrame({'id':list(test.id), 'species':list(le.inverse_transform(predicted))})

# Sample image of each species
sample_id = []
for specie in classes:
    # Pick one sample image from each species
    sample_id.append(outter_train.loc[outter_train['species'] == specie].iloc[0]['id'])
    
# Map image id with sample image
sample_table = pd.DataFrame({'id':sample_id, 'species':classes})

for i in range(1,20):
    # Random test
    random_test = random.choice(list(predicted_table.id))
    
    # Actual image from predicted_table(id)
    img_real = mpimg.imread('images_resize//'+str(random_test)+'.jpg')
        
    result_species = predicted_table.loc[predicted_table['id'] == random_test].iloc[0]['species']
    # Match species from predicted_table with reference_table to define pic of predicted species
    predict_pic = sample_table.loc[sample_table['species'] == result_species].iloc[0]['id']
    
    # Predicted image from predicted_table(species)
    img_predict = mpimg.imread('images_resize//'+str(predict_pic)+'.jpg')
        
    plt.title(result_species)
    plt.figure(i,figsize=(15,9))
    plt.subplot(121)
    plt.imshow(img_real, cmap='Set3') # show me the real leaf
    plt.subplot(122)
    plt.imshow(img_predict, cmap='Set3') # show me the predicted leaf
    plt.show()