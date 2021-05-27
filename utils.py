import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def k_fold_validator(predictor, target, vectorizer, classifier, cv=5):

    kf = KFold(n_splits=cv)
    vec = vectorizer
    clf = classifier

    train_recall_scores = []
    train_precision_scores = []
    train_f1_scores = []
    test_recall_scores = []
    test_precision_scores = []
    test_f1_scores = []
    
    print('Vectorizer:', vectorizer)
    print('Classifier:', clf)
    print('Cross-validation folds:', cv)
    
    for train_index, test_index in kf.split(predictor):

        X_tr, X_test = predictor.iloc[train_index].astype(str), predictor.iloc[test_index].astype(str)
        y_tr, y_test = target.iloc[train_index].astype(str), target.iloc[test_index].astype(str)

        X_vec_tr = vec.fit_transform(X_tr)
        X_vec_test = vec.transform(X_test)
        
        clf.fit(X_vec_tr, y_tr)

        y_pred_tr = clf.predict(X_vec_tr)
        y_pred_test = clf.predict(X_vec_test)

        train_recall_scores.append(recall_score(y_tr, y_pred_tr, pos_label='Negative emotion'))
        train_precision_scores.append(precision_score(y_tr, y_pred_tr, pos_label='Negative emotion'))
        train_f1_scores.append(f1_score(y_tr, y_pred_tr, pos_label='Negative emotion'))       
        test_recall_scores.append(recall_score(y_test, y_pred_test, pos_label='Negative emotion'))
        test_precision_scores.append(precision_score(y_test, y_pred_test, pos_label='Negative emotion'))
        test_f1_scores.append(f1_score(y_test, y_pred_test, pos_label='Negative emotion'))       
        
        plot_confusion_matrix(clf, X_vec_test, y_test)
        plt.title('Test set')
        
    print('\n')
    
    print('Train mean recall: {} +/- {}'.format(round(pd.Series(train_recall_scores).mean(), 2), 
                                               round(pd.Series(train_recall_scores).std(), 2)))
    
    print('Train mean precision: {} +/- {}'.format(round(pd.Series(train_precision_scores).mean(), 2),
                                                  round(pd.Series(train_precision_scores).std(), 2)))
    
    print('Train mean F1: {} +/- {}'.format(round(pd.Series(train_f1_scores).mean(), 2),
                                           round(pd.Series(train_f1_scores).std(), 2)))
    print('\n')
    
    print('Test mean recall: {} +/- {}'.format(round(pd.Series(test_recall_scores).mean(), 2),
                                               round(pd.Series(test_recall_scores).std(), 2)))
    
    print('Test mean precision: {} +/- {}'.format(round(pd.Series(test_precision_scores).mean(), 2),
                                                  round(pd.Series(test_precision_scores).std(), 2)))
    
    print('Test mean F1: {} +/- {}'.format(round(pd.Series(test_f1_scores).mean(), 2),
                                           round(pd.Series(test_f1_scores).std(), 2)))