

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

#print(FOLD_MAPPPING.get(0))



# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1])
# print(X)
# print(y)
# skf = StratifiedKFold(n_splits=2)
# skf.get_n_splits(X, y)


# for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

#     print(fold)
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     print('X_train:',X_train)
#     print('X_test:',X_test)



#     y_train, y_test = y[train_index], y[test_index]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(["paris", "paris", "tokyo", "amsterdam"])

le.transform(["tokyo", "tokyo", "paris"])

print(le)
