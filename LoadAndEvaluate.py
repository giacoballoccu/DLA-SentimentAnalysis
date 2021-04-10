from sklearn.metrics import plot_confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.metrics import confusion_matrix
import seaborn as sn

def confusion_matrices(true, pred, filename):
    tn, fp, fn, tp = confusion_matrix(true, pred,labels=[0, 1]).ravel()
    array = [[tp, fp], [fn, tn]]
    df_cm = pd.DataFrame(array, index = ["Pred Negative", "Pred Positive"],
                      columns = ["True Negative", "True Positive"])
    plt.figure(figsize = (10,7))
    fig = sn.heatmap(df_cm, annot=True, fmt="d")
    fig.get_figure().savefig('ConfusionMatrices/' + filename, dpi=400)

try:
    y_test = np.load("Dataset/y_test.npy")
except:
    with ZipFile('Dataset/y_test.npy.zip', 'r') as zipObj:
        zipObj.extractall('Dataset')
    y_test = np.load("Dataset/y_test.npy")

try:
  test_X = np.load("Dataset/test_X.npy")
except:
    with ZipFile('Dataset/test_X.npy.zip', 'r') as zipObj:
        zipObj.extractall('Dataset')
    test_X = np.load("Dataset/test_X.npy")

modelBDGRU = keras.models.load_model('Models/BidirectionalGRU')
print("\n BidirectionalGRU")
modelBDGRU.evaluate(test_X, y_test, batch_size=64)
confusion_matrices(np.round(modelBDGRU.predict(test_X),0), y_test, "BDGRUconfmatrix.png")

modelBDLSTM = keras.models.load_model('Models/BidirectionalLSTM')
print("\n BidirectionalLSTM")
modelBDLSTM.evaluate(test_X, y_test, batch_size=64)
confusion_matrices(np.round(modelBDLSTM.predict(test_X),0), y_test, "BDLSTMconfmatrix.png")