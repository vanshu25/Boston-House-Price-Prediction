#complete code for this project

pip install -U scikit-learn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

#GET THE DATA
boston=load_boston()
data=pd.DataFrame(data=data,columns= boston.feature_names)
data.head()

rows=2
cols=7
fig, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))

col=data.columns
index=0
for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]],ax = ax[i][j])
        index = index+1
plt.tight_layout()

corrmat = data.corr()
corrmat.index.values
def getCorrelatedFeature(corrmat,threshold):
    
    feature = []
    value  = []
    
    
    for i,index in enumerate(corrmat.index):
            
            if abs(corrmat[index])>threshold:
                feature.append(index)
                value.append(corrmat[index])
    df = pd.DataFrame(data=value,index=feature,columns=['Corr value'])
    return df
 
threshold = 0.55

corr_value = getCorrelatedFeature(corrmat['Prices'],threshold)
corr_value

correlated_data = data[corr_value.index]
correlated_data.head()

x = correlated_data.drop(labels=['Prices'],axis=1)
y = correlated_data['Prices']
x.head()

#LOGISTIC REGRESSION
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.33,random_state = 101)
model = LinearRegression()
model.fit(x_train,y_train)
from sklearn.metrics import r2_score
scores = r2_score(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)
 
    
print('r2_score:',scores)
print('mae:',mae)
print('mse:',mse)
total_features = []
total_features_name = []
selected_correlation_value =[]
r2_scores = []
mae_value = []
mse_value = []
def performance_metrics(features,th,y_true,y_pred):
    score = r2_score(y_true,y_predict)
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    
    
    total_features.append(len(features)-1)
    total_features_name.append(str(features))
    selected_correlation_value.append(th)
    r2_scores.append(scores)
    mae_value.append(mae)
    mse_value.append(mse)
    
    metrics_dataframe = pd.DataFrame(data=[total_features_name,total_features,selected_correlation_value,r2_scores,mae_value,mse_value],index=['features name','afeature','corr_value','r2 score','MAE','MSE'])
    return metrics_dataframe.T
 
 performance_metrics(correlated_data.columns.values,threshold,y_test,y_predict)
 
 
 from sklearn.model_selection import learning_curve
 def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,
                        n_jobs=None,train_sizes=np.linspace(.1,1.0,10)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("score")
    
    train_sizes,train_scores,test_scores=learning_curve(
    estimator,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X = correlated_data.drop(labels=['Prices'],axis=1)
Y = correlated_data['Prices']

title = "Learning Curves(LInear Regression)" + str(X.columns.values)
cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)

estimator = LinearRegression()
plot_learning_curve(estimator,title,X,y,ylim=(0.7,1.01),cv=cv,n_jobs=-1)
plt.show()

plt.scatter(y_test,y_predict)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

# Import SVM Regressor
from sklearn import svm

# Create a SVM Regressor
reg = svm.SVR()
reg.fit(x_train, y_train)
# Model prediction on train data
y_pred = reg.predict(x_train)
# Predicting Test data with the model
y_test_pred = reg.predict(x_test)

# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
    
 client_data = [[5,60], # Client 1
               [7,80], # Client 2
               [5,50]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
