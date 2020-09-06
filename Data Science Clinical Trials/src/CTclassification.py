# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:02:38 2019

@author: Savitri
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('..\\data\\ctclassification.csv', encoding='ISO-8859-1')
print(df.info())
#file = open('C:\\Users\\Savitri\\Documents\\ClinicalTrialdrud-deviceclassification\\results.csv','w+')

def preprocess(df):
    #df.head()
    col = ['Types','Title_condition_intervention']
    df = df[col]
    df = df[pd.notnull(df['Types'])]
    df = df[pd.notnull(df['Title_condition_intervention'])]
    df['category_id'] = df['Types'].factorize()[0]
    category_id_df = df[['Types','category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id','Types']].values)
    labels = df.category_id
 
    return id_to_category,category_to_id, labels, category_id_df
          
def tfvect(id_to_category, category_to_id, labels):
    N = 2
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Title).toarray()
    for Types, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(Types))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
 
    return features

#run the pipeline and choose the best model 
def checkmodels(labels):
    models = [
            RandomForestClassifier(n_estimators=200,max_depth=5,random_state=0),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=0)
            ]
    CV=10
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    cv_df.groupby('model_name').accuracy.mean()

    print(cv_df.groupby('model_name').accuracy.mean())
    #print(dict(cv_df.groupby('model_name').accuracy.mean()), max(cv_df.groupby('model_name').accuracy.mean()))
 
    for k, v in dict(cv_df.groupby('model_name').accuracy.mean()).items():
        if max(cv_df.groupby('model_name').accuracy.mean()) == v:
            model = k+'()'
            print(model)
    
    return cv_df, model

from sklearn.externals import joblib
import pickle

#train the data on the best chosen model
def svcmodels(df, model):
    X_train, X_test, y_train,y_test = train_test_split(df['Title_condition_intervention'],df['Types'], random_state =0)
    count_vect = CountVectorizer()
    X_train_count = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

    if model == 'LinearSVC()':
        clf = LinearSVC().fit(X_train_tfidf, y_train)
        model = LinearSVC()
        model.fit(X_train_tfidf, y_train)
        filename = 'ctclassification.sav'
        joblib.dump(model, filename)
        print('Predict CLF:=',clf.predict(count_vect.transform(["A phase 3 trial of Celgene\’s acute myeloid leukemia (AML) maintenance therapy CC-486 has met its primary endpoint. Patients who took the cytidine nucleoside analog lived longer than their peers on placebo, setting Celgene up to file for approval of the drug in the first half of next year.The trial enrolled 472 AML patients who had experienced a complete response after treatment with chemotherapy. Participants received either CC-486 or placebo orally for half of a 28-day cycle, plus best supportive care, until their disease progressed. The hope was that CC-486 would delay relapse and, in doing so, improve the survival rate in a blood cancer that kills most people within five years. Celgene now has data it thinks validates that hope, and this will please its parent Bristol-Myers Squibb, which is relying on the company to help boost its pipeline. The big biotech is yet to share data from the trial but revealed that it met its primary overall survival endpoint and key secondary objectives, including relapse-free survival. Celgene said the drug was well tolerated."])))
    elif model == 'LogisticRegression()':
        lgr = LogisticRegression().fit(X_train_tfidf, y_train)
        print('Predict LGR:=',lgr.predict(count_vect.transform(["A phase 3 trial of Celgene\’s acute myeloid leukemia (AML) maintenance therapy CC-486 has met its primary endpoint. Patients who took the cytidine nucleoside analog lived longer than their peers on placebo, setting Celgene up to file for approval of the drug in the first half of next year.The trial enrolled 472 AML patients who had experienced a complete response after treatment with chemotherapy. Participants received either CC-486 or placebo orally for half of a 28-day cycle, plus best supportive care, until their disease progressed. The hope was that CC-486 would delay relapse and, in doing so, improve the survival rate in a blood cancer that kills most people within five years. Celgene now has data it thinks validates that hope, and this will please its parent Bristol-Myers Squibb, which is relying on the company to help boost its pipeline. The big biotech is yet to share data from the trial but revealed that it met its primary overall survival endpoint and key secondary objectives, including relapse-free survival. Celgene said the drug was well tolerated."])))
    elif model == 'MultinomialNB()':
        mnb = MultinomialNB().fit(X_train_tfidf, y_train)
        print('Predict MNB:=',mnb.predict(count_vect.transform(["A phase 3 trial of Celgene\’s acute myeloid leukemia (AML) maintenance therapy CC-486 has met its primary endpoint. Patients who took the cytidine nucleoside analog lived longer than their peers on placebo, setting Celgene up to file for approval of the drug in the first half of next year.The trial enrolled 472 AML patients who had experienced a complete response after treatment with chemotherapy. Participants received either CC-486 or placebo orally for half of a 28-day cycle, plus best supportive care, until their disease progressed. The hope was that CC-486 would delay relapse and, in doing so, improve the survival rate in a blood cancer that kills most people within five years. Celgene now has data it thinks validates that hope, and this will please its parent Bristol-Myers Squibb, which is relying on the company to help boost its pipeline. The big biotech is yet to share data from the trial but revealed that it met its primary overall survival endpoint and key secondary objectives, including relapse-free survival. Celgene said the drug was well tolerated."])))
    elif model == 'RandomForestClassifier()':
        rfc = RandomForestClassifier().fit(X_train_tfidf, y_train)
        print('Predict rfc:=',rfc.predict(count_vect.transform(["A phase 3 trial of Celgene\’s acute myeloid leukemia (AML) maintenance therapy CC-486 has met its primary endpoint. Patients who took the cytidine nucleoside analog lived longer than their peers on placebo, setting Celgene up to file for approval of the drug in the first half of next year.The trial enrolled 472 AML patients who had experienced a complete response after treatment with chemotherapy. Participants received either CC-486 or placebo orally for half of a 28-day cycle, plus best supportive care, until their disease progressed. The hope was that CC-486 would delay relapse and, in doing so, improve the survival rate in a blood cancer that kills most people within five years. Celgene now has data it thinks validates that hope, and this will please its parent Bristol-Myers Squibb, which is relying on the company to help boost its pipeline. The big biotech is yet to share data from the trial but revealed that it met its primary overall survival endpoint and key secondary objectives, including relapse-free survival. Celgene said the drug was well tolerated."])))
    
    return count_vect, clf
        
import seaborn as sns

#visualize the model and their accuracy values
def visulaization(cv_df):
    fig, ax = plt.subplots(figsize=(30,30))
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()

from sklearn.metrics import confusion_matrix

#save the model 
def savemodel(features, labels,count_vect):
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.30, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
   
    # Save the vectorizer
    vec_file = 'vectorizer.pickle'
    pickle.dump(count_vect, open(vec_file, 'wb'))

    # Save the model
    mod_file = 'trailclassification.model'
    pickle.dump(model, open(mod_file, 'wb'))
    return X_test, y_test, y_pred, indices_test

# prediction heat map 
def matrix(X_test, y_test, y_pred,category_id_df):
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_df.Types.values, yticklabels=category_id_df.Types.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    plt.savefig('heatmap.png')
    return conf_mat


from IPython.display import display

def prediction(id_to_category, category_id_df,conf_mat, y_pred, y_test, indices_test):
    for predicted in category_id_df.category_id:
      for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
          print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
          display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Types', 'Title_condition_intervention']])
          print(' ')
    print('Classification Report:=',metrics.classification_report(y_test, y_pred, target_names=df['Types'].unique()))

def loadmodel(X_test, y_test,count_vect, clf):
    print('model loading ...')
    loaded_model = pickle.load(open('trailclassification.model','rb'))
    result = loaded_model.score(X_test, y_test)
    print('Score on Test data== ', result)
    test = df['Title']
    for j in test:
        print(j)
        print([j], "\t", clf.predict(count_vect.transform([j])))
        #file.write(str(loaded_model.predict(count_vect.transform([j])))
        #file.write('\n')
    
if __name__ == '__main__':
    id_to_category, category_to_id, labels, category_id_df = preprocess(df)
    features=tfvect(id_to_category, category_to_id, labels)
    cv_df, model = checkmodels(labels)
    count_vect, clf = svcmodels(df, model)
    visulaization(cv_df)
    X_test, y_test, y_pred, indices_test = savemodel(features, labels, count_vect)
    conf_mat= matrix(X_test, y_test, y_pred,category_id_df)
    prediction(id_to_category, category_id_df,conf_mat, y_pred, y_test, indices_test)
    loadmodel(X_test, y_test, count_vect, clf)












