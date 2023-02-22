

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import time


def writeCategorical(columnofDf, filename):
    uniqueValues = columnofDf.unique()
    columnofDf.value_counts()
    print(len(columnofDf.unique()))
    print(len(columnofDf.str.lower().str.strip().unique()))
    print("#######################")
    # with open(filename, 'w') as file:
    #     for uniqueValue in uniqueValues:
    #         file.write(uniqueValue + '\n')
            
            
            
#dataset loading and printing a sample of 10 rows
df = pd.read_csv('/home/gkasap/Desktop/stout_case_studyloans_full_schema.csv')
df.sample(10)

#Dataset shape -
print("The size of the dataset is", df.shape )

#get  time of the program execution
time_start = time.time()
df.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False).T

df.dropna(axis=1, thresh=int(0.5*len(df)), inplace=True)
############################################################################################################
#########find numeric columns and categorical columns#######################################################
############################################################################################################

numeric_features = df.select_dtypes(exclude='object').columns
numeric_features = numeric_features[numeric_features != 'interest_rate']
df[numeric_features].astype("float32", copy=False)
categorical_features = df.select_dtypes(include='object').columns
df["emp_title"].fillna('Unemployed', inplace=True)#
print(df[numeric_features].isna().sum())
masks = df[numeric_features].isna().any()
temp = df[numeric_features].copy()

for mask in temp.columns[masks]:
    df[mask].fillna(np.mean(df[mask]), inplace=True)

# for feature in categorical_features:
#     writeCategorical(df[feature], feature + '.txt')
df[categorical_features] = df[categorical_features].astype(str).apply(lambda x: x.str.lower()).astype(str).apply(lambda x: x.str.replace(" ", ""))
#4128-3826=302 columns less with the above line of code!!! to cna is more than once found in the same column

numericPipeline = Pipeline([('scaler', StandardScaler())])
categoricalPipeline = Pipeline([('categorical', OneHotEncoder())])

transformer = ColumnTransformer([("numeric_preprocessing", numericPipeline, numeric_features)])
transformerCategorical = ColumnTransformer([("categorical_preprocessing", categoricalPipeline, categorical_features)])               

############################################################################################################
############# split dataset to train and test and normalize the data #######################################
############################################################################################################
x_train, x_test, y_train, y_test = train_test_split(df.drop('interest_rate', axis=1), df['interest_rate'], test_size=0.2, random_state=42)

transformer.fit(x_train[numeric_features], y_train)

x_train_numeric_scaled = transformer.transform(x_train[numeric_features])
x_test_numeric_scaled = transformer.transform(x_test[numeric_features])

x_train_numeric = pd.DataFrame(x_train_numeric_scaled, columns=numeric_features, index=x_train.index)
x_train_categorical = pd.get_dummies(x_train[categorical_features])
x_train_all = pd.concat([x_train_numeric, x_train_categorical], axis=1)

x_test_numeric = pd.DataFrame(x_test_numeric_scaled, columns=numeric_features, index=x_test.index)
x_test_categorical = pd.get_dummies(x_test[categorical_features])
x_test_all = pd.concat([x_test_numeric, x_test_categorical], axis=1)


print("new length of the df with the one hot encoder {:.1f}".format(x_train_all.shape[1]))
time_end = time.time()



############################################################################################################
####PCA TIME################################################################################################
############################################################################################################




print(x_train_all.shape)
pca = PCA(n_components=0.95)
pca.fit(x_train_all[numeric_features])
print("number of components", pca.n_components_)
print("number of components", len(numeric_features))
#37-27=10 components less
feature_names = x_train_all[numeric_features].columns

feature_names = x_train_all[numeric_features].columns
selected_feature_names = [feature_names[i] for i in pca.components_[0].argsort()[::-1]]

print(pca.get_feature_names_out(x_train_all[numeric_features]))
print("program running time {:.3f}".format(time_end - time_start))


############################################################################################################
###############create a model and train it#################################################################
############################################################################################################

#random forest regressor
forest = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for forest_out_of_the_box: {mse:.7f}')
mbe=mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error for forest_out_of_the_box: {mbe:.7f}')
r2=r2_score(y_test, y_pred)
print(f'r2 for forest_out_of_the_box: {r2:.7f}')


#TRY WITH NEURAL NETWORK!!!!!!!!!!! moglare

















# df.columns[df.isnull().any()].tolist()
# print("The columns that have null values are",len(df.columns[df.isnull().any()].tolist()), "and are", df.columns[df.isnull().any()].tolist())

# df.isnull().sum()

# print( "The total number of null values are", df.isnull().sum().sum()) 
# freq=df.isnull().sum().sum()/(df.shape[0]*df.shape[1])
# print(f"The total percentage of null values are {freq * 100:.2f} %. " )

# Visualize missing values as a matrix
# msno.matrix(df, labels = True,)

# plt.title('Null values visual bar plot',fontdict={'fontsize': 20})
# msno.bar(df)

#Dataframe that represents the percentage of columns with missing values
# prc_missing_values = pd.DataFrame()


#Columns with high percentage of null values may cause trouble so we remove them based on a threshold of our desire
#threshold choice= 0.5
#new dataframe named af for dataset after columns drop
# af=df.dropna(axis=1, thresh=int(0.5*len(df)))
# af.shape

#Null check again
