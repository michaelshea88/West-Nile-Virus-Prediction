import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
import warnings
warnings.filterwarnings("ignore")
# define a random state and train test split % to use throughout
rs = 8
tts = .2
# define functions for classification model evaluation
def doClassifMetrics(y_test, y_pred):
    print 'Confusion Matrix \n', confusion_matrix(y_test, y_pred)
    print 'Classification Report \n', classification_report(y_test, y_pred)

def modelEval(name, model, X, y, binarize_threshold):
    X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.2,
        stratify = y, random_state = rs)
    meancvscore = cross_val_score(model, X, y, n_jobs=-1, verbose=1).mean()
    print 'Model %s cross_val_score: %f' % (name, meancvscore)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_adj = binarize(model.predict_proba(X_test)[:,1],
        threshold = binarize_threshold, copy=False).transpose()
    print 'Model %s classification metrics:' % name
    doClassifMetrics(y_test, y_pred)
    print 'Model %s using prediction threshold %f:' % (name, binarize_threshold)
    doClassifMetrics(y_test, y_pred_adj)

# import data and define X, y (features, target)
df = pd.read_csv('../Assets/merged.csv')
df.info()
X = df.drop(['Date', 'Trap', 'Year','WnvPresent', 'date_station_id'], axis=1)
y = df['WnvPresent'].astype(int)

# select the 10 best features
kbest = SelectKBest(k=10)
X_kbest = kbest.fit_transform(X,y)

# get list of best feature names, and put into a dataframe for reference
best_features = [x for (x,y) in zip(X.columns, kbest.get_support().tolist()) if y==1]
X_kbest = pd.DataFrame(X_kbest, columns = best_features)



# instantiate and fit LogisticRegression
lg = LogisticRegression(random_state=rs, n_jobs=-1)

modelEval('LogisticRegression', lg, X_kbest, y, 0.2)


# instantiate  RandomForestClassifier
rf = RandomForestClassifier(random_state=rs, n_jobs=-1, max_depth = None)

modelEval('RandomForestClassifier', rf, X_kbest, y, 0.2)
