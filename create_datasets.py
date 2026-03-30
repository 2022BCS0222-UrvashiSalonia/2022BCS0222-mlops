import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Version 1 - partial dataset (first 100 rows, 3 features only)
df_v1 = df.iloc[:100][['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'target']]
df_v1.to_csv('data/iris_v1.csv', index=False)
print("V1 created:", df_v1.shape)

# Version 2 - full dataset (all 150 rows, all features)
df_v2 = df.copy()
df_v2.to_csv('data/iris_v2.csv', index=False)
print("V2 created:", df_v2.shape)
