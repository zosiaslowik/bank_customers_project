import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score, roc_curve
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

warnings.filterwarnings("ignore")

data = pd.read_csv('BankCustomerData.csv')

pd.set_option('display.max_columns', 40)

#wielkosci statystyczne opisujace dane: srednia, min i max, odchylenie
description = data.describe()
print("WIELKOŚCI STATYSTYCZNE OPISUJĄCE DANE: ")
print(description, "\n\n")

# ile duplikatow
print("LICZBA DUPLIKATÓW: ", data.duplicated().sum(),"\n\n")

print("ILE NIEZNANYCH WARTOŚCI W KOLUMNIE")
for col in data:
    print(col, ": ", (data[col] == 'unknown').sum())
print("\n\n")

# wartosci w kolumnie
print("WARTOŚCI W KOLUMNIE")
for col in data:
    if data[col].dtype == object:
        dat = list(data[col].unique())
        dat.sort()
        print(col, ": ", dat)
print("\n\n")

data.drop(columns=['poutcome'], inplace=True)
data = data[~(data == 'unknown').any(axis=1)]

colors = ['peachpuff', 'lightgreen', 'lightblue', 'lightpink']



def to_num(value):
    if value == 'no':
        return 0
    elif value == 'yes':
        return 1


def contact_to_num(value):
    if value == 'cellular':
        return 0
    elif value == 'telephone':
        return 1


data['default'] = data['default'].apply(to_num)
data['loan'] = data['loan'].apply(to_num)
data['housing'] = data['housing'].apply(to_num)
data['term_deposit'] = data['term_deposit'].apply(to_num)
data['contact'] = data['contact'].apply(contact_to_num)
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'month'])

# wartosci w kolumnie
print("WARTOŚCI W KOLUMNIE PO ZAMIANIE")
for col in data:
    dat = list(data[col].unique())
    dat.sort()
    print(col, ": ", dat)
print("\n\n")

features = [col for col in data if col != 'term_deposit']
data_to_stand = data[features]
scaler = StandardScaler()
data_to_stand = scaler.fit_transform(data_to_stand)

data_stand = pd.DataFrame(data_to_stand, columns=features)
data_target = data['term_deposit']
data_target = data_target.reset_index(drop=True)
ros = RandomOverSampler(sampling_strategy="not majority")

data = pd.concat([data_stand, data_target], axis=1)
minority_df = data[data['term_deposit'] == 1]
majority_df = data[data['term_deposit'] == 0]

data, test = train_test_split(data, train_size=0.8)

target_train = data['term_deposit']
data.drop(columns=['term_deposit'], inplace=True)
data, target_train = ros.fit_resample(data, target_train)

target_test = test['term_deposit']
test.drop(columns=['term_deposit'], inplace=True)

knn = KNeighborsClassifier()
knn.fit(data, target_train)
target_knn = knn.predict(test)
recall_knn = recall_score(target_test, target_knn)
print('RECALL KNN: ')
print(recall_knn)
con_mat_knn = confusion_matrix(target_test, target_knn)
print('TABLICA POMYŁEK KNN: ')
print(con_mat_knn)
precision_knn = precision_score(target_test, target_knn)
print('PRECISION KNN: ')
print(precision_knn)
f1_score_knn=f1_score(target_test, target_knn)
print('F1 SCORE KNN: ')
print(f1_score_knn)
print('ACCURACY SCORE KNN: ')
accuracy_knn=accuracy_score(target_test, target_knn)
print(accuracy_knn, "\n")

dt = DecisionTreeClassifier()
dt.fit(data, target_train)
target_dt = dt.predict(test)

recall_dt = recall_score(target_test, target_dt)
print('RECALL DT: ')
print(recall_dt)
con_mat_dt = confusion_matrix(target_test, target_dt)
print('TABLICA POMYŁEK DT: ')
print(con_mat_dt)
precision_dt = precision_score(target_test, target_dt)
print('PRECISION DT: ')
print(precision_dt)
f1_score_dt = f1_score(target_test, target_dt)
print('F1 SCORE DT: ')
print(f1_score_dt)
print('ACCURACY SCORE DT: ')
accuracy_dt = accuracy_score(target_test, target_dt)
print(accuracy_dt, "\n")



nb = GaussianNB()
nb.fit(data, target_train)
target_nb = nb.predict(test)

recall_nb = recall_score(target_test, target_nb)
print('RECALL NB: ')
print(recall_nb)
con_mat_nb = confusion_matrix(target_test, target_nb)
print('TABLICA POMYŁEK NB: ')
print(con_mat_nb)
precision_nb = precision_score(target_test, target_nb)
print('PRECISION NB: ')
print(precision_nb)
f1_score_nb = f1_score(target_test, target_nb)
print('F1 SCORE NB: ')
print(f1_score_nb)
print('ACCURACY SCORE NB: ')
accuracy_nb = accuracy_score(target_test, target_nb)
print(accuracy_nb, "\n")



gbc = GradientBoostingClassifier()
gbc.fit(data, target_train)
target_gbc = gbc.predict(test)
recall_gbc = recall_score(target_test, target_gbc)
print('RECALL GBC: ')
print(recall_gbc)
con_mat_gbc = confusion_matrix(target_test, target_gbc)
print('TABLICA POMYŁEK GBC: ')
print(con_mat_gbc)
precision_gbc = precision_score(target_test, target_gbc)
print('PRECISION GBC: ')
print(precision_gbc)
f1_score_gbc = f1_score(target_test, target_gbc)
print('F1 SCORE GBC: ')
print(f1_score_gbc)
print('ACCURACY SCORE GBC: ')
accuracy_gbc = accuracy_score(target_test, target_gbc)
print(accuracy_gbc, "\n")

rf = RandomForestClassifier()
rf.fit(data, target_train)
target_rf = rf.predict(test)
recall_rf = recall_score(target_test, target_rf)
print('RECALL RF: ')
print(recall_rf)
con_mat_rf = confusion_matrix(target_test, target_rf)
print('TABLICA POMYŁEK RF: ')
print(con_mat_rf)
precision_rf = precision_score(target_test, target_rf)
print('PRECISION RF: ')
print(precision_rf)
f1_score_rf = f1_score(target_test, target_rf)
print('F1 SCORE RF: ')
print(f1_score_rf)
print('ACCURACY SCORE RF: ')
accuracy_rf = accuracy_score(target_test, target_rf)
print(accuracy_rf, "\n")


svm = SVC()
svm.fit(data, target_train)
target_svm = svm.predict(test)
recall_svm = recall_score(target_test, target_svm)
print('RECALL SVM: ')
print(recall_svm)
con_mat_svm = confusion_matrix(target_test, target_svm)
print('TABLICA POMYŁEK SVM: ')
print(con_mat_svm)
precision_svm = precision_score(target_test, target_svm)
print('PRECISION SVM: ')
print(precision_svm)
f1_score_svm = f1_score(target_test, target_svm)
print('F1 SCORE SVM: ')
print(f1_score_svm)
print('ACCURACY SCORE SVM: ')
accuracy_svm = accuracy_score(target_test, target_svm)
print(accuracy_svm, "\n")


labels = ['KNN', 'DT', 'NB', 'GBC', 'RF', 'SVM']
recall_scores = [recall_knn, recall_dt, recall_nb, recall_gbc, recall_rf, recall_svm]
precision_scores = [precision_knn, precision_dt, precision_nb, precision_gbc, precision_rf, precision_svm]
f1_scores = [f1_score_knn, f1_score_dt, f1_score_nb, f1_score_gbc, f1_score_rf, f1_score_svm]
accuracy_scores = [accuracy_knn, accuracy_dt, accuracy_nb, accuracy_gbc, accuracy_rf, accuracy_svm]

colors = ['peachpuff', 'lightgreen', 'lightblue', 'lightpink']

fig, ax = plt.subplots()
width = 0.2

rects_recall = ax.bar(np.arange(len(labels)) - width, recall_scores, width, label='Recall', color=colors[0])
rects_precision = ax.bar(np.arange(len(labels)), precision_scores, width, label='Precision', color=colors[1])
rects_f1 = ax.bar(np.arange(len(labels)) + width, f1_scores, width, label='F1 Score', color=colors[2])
rects_accuracy = ax.bar(np.arange(len(labels)) + 2 * width, accuracy_scores, width, label='Accuracy', color=colors[3])


ax.set_title('Metryki')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects_recall)
autolabel(rects_precision)
autolabel(rects_f1)
autolabel(rects_accuracy)

plt.tight_layout()
plt.show()
