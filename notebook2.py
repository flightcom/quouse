import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

# Read in train data
# df_train = pd.read_csv('./data/kaggle/house-prices/train.csv', index_col=0)
df_train = pd.read_csv('./data/etalab/train.csv', index_col=0)
df_train = df_train.sample(frac=1).reset_index(drop=True)

print(df_train.head())

# print(df_train['SalePrice'].describe())

# #histogram
# sns.distplot(df_train['SalePrice']);

# #skewness and kurtosis
# print("Skewness: %f" % df_train['SalePrice'].skew())
# print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# #correlation matrix
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);

# #saleprice correlation matrix
# k = 10 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# #scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size = 2.5)
# plt.show();

# #missing data
# total = df_train.isnull().sum().sort_values(ascending=False)
# percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# missing_data.head(20)

# df_train = df_train.fillna(df_train.mean())

# #standardizing data
# saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
# low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
# high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)

# #bivariate analysis saleprice/grlivarea
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# cols = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
# cols = ['city', 'date_mutation', 'latitude', 'longitude', 'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local', 'valeur_fonciere']
cols = ['city', 'latitude', 'longitude', 'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local', 'valeur_fonciere']
# cols = ['nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local', 'valeur_fonciere']
df_train = df_train[cols]
# Create dummy values
df_train = pd.get_dummies(df_train)
#filling NA's with the mean of the column:
df_train = df_train.fillna(df_train.mean())
print(df_train.head())
# Always standard scale the data before using NN
scale = StandardScaler()
# scale = MinMaxScaler()
# X_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]
X_train = df_train[['city', 'latitude', 'longitude', 'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local']]
# X_train = df_train[['nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local']]
X_train = scale.fit_transform(X_train)
# Y is just the 'SalePrice' column
y = df_train['valeur_fonciere'].values
seed = 7
np.random.seed(seed)
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.33, random_state=seed)

epochs=150

def create_model():
    # create model
    model = Sequential()
    # model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, input_dim=X_train.shape[1]))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(70, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.25))
    # model.add(Dense(70, activation='relu'))
    model.add(Dense(1))
    # create optimizer
    learning_rate=0.0005
    decay_rate = learning_rate / epochs
    momentum = 0.8
    optimizer = optimizers.SGD(lr=learning_rate, clipnorm=1)
    optimizer = optimizers.RMSprop(lr=learning_rate)
    # optimizer = optimizers.Adam(learning_rate=0.0003, beta_1=0.009, beta_2=0.0099, amsgrad=True)
    # optimizer = optimizers.Nadam(learning_rate=0.0003, beta_1=0.009, beta_2=0.0099)
    # optimizer = optimizers.Adagrad(learning_rate=0.0003)

    # Compile model
    # model.compile(optimizer ='adam', loss = 'mean_squared_error', 
    #           metrics =[metrics.mae])
    model.compile(optimizer = optimizer, loss = 'mse', 
              metrics = [metrics.mae])

    return model

model = create_model()
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=epochs, batch_size=32)

# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# # summarize history for learning rate
# plt.plot(history.history['lr'])
# plt.title('learning rate')
# plt.ylabel('learning rate')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# df_test = pd.read_csv('./data/kaggle/house-prices/test.csv')
df_test = pd.read_csv('./data/etalab/test.csv', index_col=0)
# cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
cols = ['city', 'date_mutation', 'latitude', 'longitude', 'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local']
# cols = ['nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local']
id_col = df_test['id'].values.tolist()
# df_test['GrLivArea'] = np.log1p(df_test['GrLivArea'])
df_test = pd.get_dummies(df_test)
df_test = df_test.fillna(df_test.mean())
print(df_test.head())

X_test = df_test[cols].values
# Always standard scale the data before using NN
scale = StandardScaler()
# scale = MinMaxScaler()
X_test = scale.fit_transform(X_test)

prediction = model.predict(X_test)

submission = pd.DataFrame()
submission['id'] = id_col
submission['valeur_fonciere'] = prediction

submission.to_csv('./data/etalab/submission.csv', index=False)
