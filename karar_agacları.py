from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Verilerimizi oluşturduk
X_train = np.array([
    [130, 30, 10],
    [125, 36, 11],
    [135, 34, 10],
    [133, 30, 9],
    [129, 38, 12],
    [180, 90, 30],
    [190, 80, 25],
    [175, 90, 35],
    [177, 60, 22],
    [185, 105, 33],
    [165, 55, 27],
    [155, 50, 44],
    [160, 58, 39],
    [162, 59, 41],
    [167, 62, 55],
    [174, 70, 47],
    [193, 90, 23],
    [187, 80, 27],
    [183, 88, 28],
    [159, 40, 29],
    [164, 66, 32],
    [166, 56, 42]
])

y_train = ['e', 'e', 'k', 'k', 'e', 'e', 'e', 'e', 'k', 'e', 'k', 'k', 'k', 'k', 'k', 'e', 'e', 'e', 'e', 'k', 'k', 'k']

# Decision Tree modelini oluşturup ve eğiticez
tree_classifier = DecisionTreeClassifier(random_state=0)
tree_classifier.fit(X_train, y_train)

# Test verisi oluştur (Örneğin, yeni bir kişi)
X_test = np.array([
    [150, 50, 20]
])

# Tahmin yapın
prediction = tree_classifier.predict(X_test)

print("Tahmin:", prediction)




