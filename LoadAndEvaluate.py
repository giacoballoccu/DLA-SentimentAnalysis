import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
from joblib import load
import time

def confusion_matrices(true, pred, filename):
    tn, fp, fn, tp = confusion_matrix(true, pred,labels=[0, 1]).ravel()
    array = [[tp, fp], [fn, tn]]
    df_cm = pd.DataFrame(array, index = ["Pred Negative", "Pred Positive"],
                      columns = ["True Negative", "True Positive"])
    plt.figure(figsize = (10,7))
    fig = sn.heatmap(df_cm, annot=True, fmt="d")
    fig.get_figure().savefig('ConfusionMatrices/' + filename, dpi=400)

test_X_baseline = np.load("Dataset/test_X_baseline.npy")
y_test_baseline = np.load("Dataset/y_test_baseline.npy")

try:
  test_X_ANN = np.load("Dataset/test_X_ANN.npy")
except:
    with ZipFile('Dataset/test_X_ANN.npy.zip', 'r') as zipObj:
        zipObj.extractall('Dataset')
    test_X_ANN = np.load("Dataset/test_X_ANN.npy")

y_test_ANN = np.load("Dataset/y_test_ANN.npy")

y_test_ANN = y_test_ANN.astype('bool')
y_test_baseline = y_test_baseline.astype('bool')

print("Evaluating SVM ...\n")
svm = load("Models/svm.joblib")
start_time = time.time()
y_pred = svm.predict(test_X_baseline)
print("Prediction time for SVM: %d seconds - accuracy: %s" % ((time.time() - start_time), accuracy_score(y_test_baseline, y_pred)))
print("\n")

print("Evaluating Random forest ...\n")
rf = load("Models/randomforest.joblib")
start_time = time.time()
y_pred = rf.predict(test_X_baseline)
print("Prediction time for Random forest: %d seconds - accuracy: %s" % ((time.time() - start_time), accuracy_score(y_test_baseline, y_pred)))
print("\n")


print("Evaluating BDRNN GRU ...\n")
start_time = time.time()
modelBDGRU = keras.models.load_model('Models/BidirectionalGRU')
modelBDGRU.evaluate(test_X_ANN, y_test_ANN, batch_size=64)
print("Prediction time for BDRNN GRU: %s seconds\n" % (time.time() - start_time))

print("Evaluating BDRNN LSTM ...\n")
start_time = time.time()
modelBDLSTM = keras.models.load_model('Models/BidirectionalLSTM')
print("\n BidirectionalLSTM")
modelBDLSTM.evaluate(test_X_ANN, y_test_ANN, batch_size=64)
print("Prediction time for BDRNN LSTM: %s seconds\n" % (time.time() - start_time))
