#veri temizliği
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#!pip install seaborn

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\Casper\PycharmProjects\projects\penguins_size.csv")
df.head()

df.info()

print(df.shape)
#Değer sayısı ve kolon sayısını gösterir.

#df.describe()
#sayısal olan tüm değerleri getirir

df.describe(include='all')
#sayısal ve sayısal olmayan tüm kolonları getirir.

df.corr()
#korelasyon haritasını gösterir.

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

#df.isna().sum()
#eksik değerleri gösterir.
#df.isna().sum() / df.count()
# eksik değerleri kolon sayısına bölüyor.
#df.isna().sum() / df.count() * 100
# eksik değerleri kolon sayısına bölüp 100 ile çarpıyor.

nan_percentage = df.isna().sum() / df.count() * 100
nan_count = df.isna().sum()
nan_count

nan_table = pd.concat([nan_count, nan_percentage], axis=1)
nan_table.columns = ['Count', 'Percantage']
nan_table

#!pip install scikit-learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
# df.iloc[:, :]
# bütün satırlar bütün sütünları getir demek.
df.iloc[:, :] = imputer.fit_transform(df)
#boşlukları kolon bazında en çok tekrarlanan değerle tamamlar.

df.isna().sum()
df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['sex'])
df['sex'].value_counts()
df.head()

df = df.drop(labels=['sex'], axis=1)
df.head()

species_count = df['species'].value_counts().reset_index()
species_count

sns.barplotxlabel(data=species_count, x='index', y='species')
plt.show()

df[df['species']=='Adelie']['body_mass_g']
# sns.kdeplot(df[df['species']=='Adelie']['body_mass_g'])
# sns.kdeplot(df[df['species']=='Gentoo']['body_mass_g'])
# sns.kdeplot(df[df['species']=='Chinstrap']['body_mass_g'])

#for spec in df['species'].unique():
#    sns.kdeplot(df[df['species'] ==spec]['body_mass_g'], shade=True, label=spec)
#    plt.legend()
# plt.show()

for col in df.columns[2:6]:
    print(col)

for col in df.columns[2:6]:
    for spec in df['species'].unique():
        sns.kdeplot(df[df['species'] == spec]['body_mass_g'], shade=True, label=spec)
        plt.legend()
    plt.show()

sns.pairplot(df, hue='species', size=3, diag_kind='hist')
plt.show()

## Machine Learning
df.head()

# pd.get_dummies(df[['island']])
# onehotencoding yapmanın kısa yolu
island = pd.get_dummies(df[['island']], drop_first=True)
df2 = pd.concat([df, island], axis=1).drop(['island'], axis=1)
df2.head()

target, features = df2.species, df2.drop('species', axis=1)

from sklearn.preprocessing import StandardScaler
features.head()
scaler = StandardScaler()

scaler.fit(features.iloc[:, :4])

features.iloc[:, :4]  = scaler.transform(features.iloc[:, :4])
features

target

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
target_encoded = le.fit_transform(target)
target_encoded

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2,  random_state=41)
x_train.shape
x_test.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
predict = tree.predict(x_test)
confusion_matrix(y_test, predict)

sns.heatmap(confusion_matrix(y_test, predict))
plt.show()

print(classification_report(y_test, predict))

print('Accuracy Score= %', accuracy_score(y_test, predict) * 100)

final_df = pd.concat([features, pd.Series(target_encoded, name='Target')], axis=1)

final_df.to_csv('penguins_finish.csv', index=False)
#temizleme işlemi yapılan csv dosyasını yeni bir csv dosyası haline getiriypr.

