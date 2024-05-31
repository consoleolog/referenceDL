import tensorflow as tf
import pandas as pd
import numpy as np


data = pd.read_csv('data/train.csv')

# 최빈값
embarked_mode = data['Embarked'].mode()

age_avg = data['Age'].mean()

data['Age'].fillna(round(age_avg), inplace=True)
data['Embarked'].fillna(value='S', inplace=True)

# print(data['Age'].isnull().sum())
# print(data['Embarked'].isnull().sum())
print(type(np.array(data)))

# 정답 분리
정답 = data.pop('Survived')
print(type(정답))
exit()
# 데이터와 정답을 함께 포함하는 딕셔너리 형태로 변환
ds = tf.data.Dataset.from_tensor_slices((dict(data), 정답))
# print(ds.take(1))
# for i, j in ds.take(1):
#     print(i, j)

# 피처 걸럼
feature_columns = []

test_feature_columns = []

Fare = tf.feature_column.numeric_column('Fare')
Parch = tf.feature_column.numeric_column('Parch')
SibSp = tf.feature_column.numeric_column('SibSp')

test_data = pd.read_csv('data/test.csv')


feature_columns.append(Fare)
feature_columns.append(Parch)
feature_columns.append(SibSp)

Age = tf.feature_column.numeric_column('Age')
Age_bucket = tf.feature_column.bucketized_column(Age, boundaries=[10, 20, 30, 40, 50, 60])
feature_columns.append(Age_bucket)

Sex_vocab = data['Sex'].unique()
Sex_cat = tf.feature_column.categorical_column_with_vocabulary_list('Sex', Sex_vocab)
Sex_one_hot = tf.feature_column.indicator_column(Sex_cat)
feature_columns.append(Sex_one_hot)

Embarked_vocab = data['Embarked'].unique()
Embarked_cat = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', Embarked_vocab)
Embarked_one_hot = tf.feature_column.indicator_column(Embarked_cat)
feature_columns.append(Embarked_one_hot)

Pclass_vocab = data['Pclass'].unique()
Pclass_cat = tf.feature_column.categorical_column_with_vocabulary_list('Pclass',Pclass_vocab)
Pclass_one_hot = tf.feature_column.indicator_column(Pclass_cat)
feature_columns.append(Pclass_one_hot)

#embedding
Ticket_vocab = data['Ticket'].unique()
Ticket_cat = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', Ticket_vocab)
Ticket_one_hot = tf.feature_column.embedding_column(Ticket_cat, dimension=9)
feature_columns.append(Ticket_one_hot)

#ds_batch = ds.batch(32)
# feature_layer = tf.keras.layers.DenseFeatures(tf.feature_column.numeric_column('Fare'))
# feature_layer(next(iter(ds_batch))[0])
# exit()

model = tf.keras.models.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

ds_batch = ds.batch(32)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(ds_batch, epochs=100, shuffle=True)

a = []



model.predict()
# 숫자로 집어 넣을거(numeric_column) Fare Parch SibSp

# 뭉퉁 그려서 집어 넣을거 (bucketized_column) Age

# 종류 몇개 없는 카테 고리 화해서 집어 넣을거 (indicator_column)Sex   Embarked    Pclass

# 종류 많은 카테 고리 (embedding_colum)   Ticket