import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from joblib import dump, load
from mufasa_model import mufasa, mufasa_classifier, CNN_1d
from random import randrange
from random import sample

X = pkl.load(open("feature_matrix.pkl", "rb"))
y = pkl.load(open("label_matrix.pkl", "rb"))

a_list = []  # list of models' accuracy
p_list = []  # list of models' precision
r_list = []  # list of models' recall
f_list = []  # list of models' f1-score

k = 10
flip_prob = 0.03
for i in range(k):
    print('---------> ' + str(i))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=randrange(0, 100))

    # --- injecting noisy labels --- #
    num_training_labels = len(y_train)
    expected_flips = int(num_training_labels * flip_prob)
    flip_index = sample(range(0, num_training_labels-1), expected_flips)
    for idx in flip_index:
        if y_train[idx] == 0:
            y_train[idx] = 1
        else:
            y_train[idx] = 0

    # --- Option 1: MuFasa --- #
    # mf = mufasa_classifier()
    # mf.train_mufasa(X_train, y_train)
    # score, y_pred = mf.test_mufasa(X_test, y_test, thre = 0.5)

    # --- Option 2: MLP --- #
    clf = MLPClassifier(hidden_layer_sizes=(32, 2), activation='relu').fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    f = f1_score(y_test, y_pred, average=None)
    p = precision_score(y_test, y_pred, average=None)
    r = recall_score(y_test, y_pred, average=None)
    a_list.append(score)
    f_list.append(f)
    p_list.append(p)
    r_list.append(r)

print("accuracy: " + str(np.average(a_list)) + u" \u00B1 " + str(np.std(a_list)))
print("precision: " + str(np.average(p_list)) + u" \u00B1 " + str(np.std(p_list)))
print("recall: " + str(np.average(r_list)) + u" \u00B1 " + str(np.std(r_list)))
print("f1-score: " + str(np.average(f_list)) + u" \u00B1 " + str(np.std(f_list)))

