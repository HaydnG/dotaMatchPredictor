import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


# df = pd.read_csv('data.csv')
#
# Y = df['RadiantWin']
# del df['RadiantWin']
# X = df
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
#
#
# model = Sequential()
# model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
#
# model.fit(X_train, y_train, epochs=300, batch_size=32)
#
# y_hat = model.predict(X_test)
# y_hat = [0 if val < 0.5 else 1 for val in y_hat]
#
# print(accuracy_score(y_test, y_hat))
#
# model.save('dotaPre')

model = load_model('dotaPre')

#df = pd.DataFrame([[35,70,18,25,83,28,8,12,73,105]])

df = pd.DataFrame([[14,129,17,32,111,97,57,123,109,9]])
y = model.predict(df)

print(y)

#https://www.youtube.com/watch?v=6_2hzRopPbQ
#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough