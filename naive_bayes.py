#kütüphaneler yüklendi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri seti yüklendi
veriler = pd.read_csv('veriler.csv')

#bağımsız ve bağımlı değişkenler belirlendi
x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolündü
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklendi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#naive bayes kütüphanesi import edildi ve tahmin yaptırıldı
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

#confusion metrix kütüphanesi eklendi ve eğitimin başarısı yazdırıldı
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)
