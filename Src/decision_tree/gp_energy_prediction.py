
# coding: utf-8

# In[1]:

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# # Feature scaling
# - Would numerical feature scaling make a difference in this problem? How should we do it? What is Z-score? 

# In[2]:

datadir='../APPL/data/shift_freq/'


# In[54]:

ggstr="1.0"


# In[55]:

x1=np.loadtxt(datadir+"potential-g_"+ggstr+"_.dat")
x2=np.loadtxt(datadir+"dens-g_"+ggstr+"_.dat")
y0=np.loadtxt(datadir+"energy-g_"+ggstr+"_.dat")
y1=np.loadtxt(datadir+"ekin-g_"+ggstr+"_.dat")
y2=np.loadtxt(datadir+"epot-g_"+ggstr+"_.dat")
y3=np.loadtxt(datadir+"eint-g_"+ggstr+"_.dat")


# In[56]:

type(x1)


# In[57]:

y1.shape


# In[58]:

X=np.concatenate((x1,x2),axis=1)
y=y0
y.reshape(50000,1)


# In[59]:

y=np.concatenate((y0,y1,y2,y3))
y=y.reshape(4,50000).transpose()


# In[60]:

plt.plot(x2[-1])
plt.plot(x2[-2])


# In[61]:

f, ax = plt.subplots(1,4,figsize=(16, 4))
titles=["e_tot","e_kin","e_pot","e_int"]
for i in range(4):
    ax[i].hist(y[:,i])
    ax[i].set_title(titles[i])


# ## Decision Tree

# In[71]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[:9999,:], y[:9999,:], test_size=0.2, random_state=42)


# In[72]:

y_train.shape


# In[73]:

from sklearn.tree import DecisionTreeRegressor


# In[74]:

#tree_reg = DecisionTreeRegressor(max_depth=35, random_state=42)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)


# In[75]:

y_pred = tree_reg.predict(X_test)


# In[76]:

f, ax = plt.subplots(1,4,figsize=(16, 4))
titles=["e_tot","e_kin","e_pot","e_int"]
for i in range(4):
    ax[i].plot(y_test[:,i],y_pred[:,i],'.')
    ax[i].plot(y_test[:,i],y_test[:,i],'r-')
    ax[i].set_title(titles[i])


# In[77]:

tree_reg.score(X_test,y_test)


# In[78]:

np.sum((y_pred-y_test)**2, 0)/y_test.shape[0]


# In[79]:

np.sum((y_pred-y_test)**2/y_test**2, 0)/y_test.shape[0]


# # 5-fold cross validation

# In[38]:

from sklearn.model_selection import cross_val_score


# In[65]:

#%timeit 
cv_results = cross_val_score(tree_reg, X, y, cv=5)


# In[66]:

print(cv_results)


# # Cross validation randomized search

# In[37]:

# Import necessary modules
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 128)}#,
              #"min_samples_leaf": randint(1, 9)}#,
              #"criterion": ["gini", "entropy"]}


# In[39]:

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree_reg, param_dist, cv=5)


# In[77]:

tree_cv.fit(X_train,y_train)


# In[78]:

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[79]:

y_pred=tree_cv.predict(X_test)


# In[80]:

# Predict on the test set and compute metrics
from sklearn.metrics import mean_squared_error
y_pred = tree_cv.predict(X_test)
r2 = tree_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned Tree params: {}".format(tree_cv.best_params_))
print("Tuned Tree R squared: {}".format(r2))
print("Tuned Tree MSE: {}".format(mse))


# ## Kernel Ridge Regression

# In[122]:

from sklearn.kernel_ridge import KernelRidge


# In[158]:

kridge = KernelRidge(alpha=.5,degree=5,kernel='rbf')
#kridge = KernelRidge(alpha=1.0,degree=5,kernel='polynomial')
#kridge = KernelRidge(alpha=0.1,degree=10,kernel='linear')
kridge.fit(X_train[0:10000,:], y_train[0:10000]) 


# In[159]:

y_pred_KRR = kridge.predict(X_test)


# In[160]:

f, ax = plt.subplots(1,4,figsize=(16, 4))
titles=["gg","e_int","e_kin","e_pot"]
for i in range(4):
    ax[i].plot(y_test[:,i],y_pred_KRR[:,i],'.')
    ax[i].plot(y_test[:,i],y_test[:,i],'r-')
    ax[i].set_title(titles[i])


# In[161]:

score_KRR=kridge.score(X_test,y_test)


# In[162]:

score_KRR


# In[45]:

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 10)
kridge_scores = []
kridge_scores_std = []


# In[46]:

for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    kridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    kridge_cv_scores = cross_val_score(kridge,X,y,cv=5)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    kridge_scores.append(np.mean(kridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    kridge_scores_std.append(np.std(kridge_cv_scores))


# In[47]:

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# In[48]:

# Display the plot
display_plot(kridge_scores, kridge_scores_std)


# # XGboost

# In[23]:

import xgboost as xgb


# In[27]:

x1=np.loadtxt("nonlinearSE/potential-g.dat")
x2=np.loadtxt("nonlinearSE/dens-g.dat")
y0=np.loadtxt("nonlinearSE/gg-g.dat")
y1=np.loadtxt("nonlinearSE/eint-g.dat")
y2=np.loadtxt("nonlinearSE/ekin-g.dat")
y3=np.loadtxt("nonlinearSE/epot-g.dat")


# In[28]:

X=np.concatenate((x1,x2),axis=1)
y=y0


# In[29]:

#y=np.concatenate((y0,y1,y2,y3))
#y=y.reshape(4,8481).transpose()


# In[30]:

plt.plot(x2[-1])
plt.plot(x2[-2])


# In[32]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# In[34]:

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations


# In[37]:

from sklearn.metrics import mean_squared_error


# In[44]:

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective="reg:linear",n_estimators=10, seed=123)

# Fit the regressor to the training set
xg_reg.fit(X_train,y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

xg_reg.score(X_test,y_test)


# In[45]:

# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(X_train, y_train)
DM_test =  xgb.DMatrix(X_test, y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


# ## XGboost cross validation

# In[48]:

# Create the DMatrix: housing_dmatrix
gp_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]))


# In[54]:

# Create the DMatrix: housing_dmatrix
gp_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="mae", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))


# ### Xgboost regularization

# In[53]:

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:linear","max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))


# ### Visualizing trees

# In[59]:

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=gp_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg,num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg,num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg,num_trees=9,rankdir="LR")
plt.show()


# ### plot important features

# In[67]:

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Train the model: xg_reg
xg_reg=xgb.train(params=params, dtrain=gp_dmatrix, num_boost_round=10)

# Plot the feature importances
ax=xgb.plot_importance(xg_reg)
fig = ax.figure
fig.set_size_inches(12,20)
plt.show()


# ## Tuning xgboost parameters

# In[68]:

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:linear", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))


# #### early stopping: stop boost when error does not improve

# In[70]:

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=3, early_stopping_rounds=10, num_boost_round=100, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)


# #### learning rate

# In[72]:

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:linear", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta 
for curr_val in eta_vals:

    params["eta"] = curr_val

# Perform cross-validation with early stopping: cv_results
    cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=3, early_stopping_rounds=5, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))


# #### max_depth

# In[73]:

# Create the parameter dictionary
params = {"objective":"reg:linear"}

# Create list of max_depth values
max_depths = [2,5,10,20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=2, early_stopping_rounds=5, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))


# #### colsample_by_tree

# In[76]:

# Create the parameter dictionary
params={"objective":"reg:linear","max_depth":3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.2,0.4,0.6,0.8,1.0]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:

    params["colsample_bytree"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=gp_dmatrix, params=params, nfold=5,
                 num_boost_round=50, early_stopping_rounds=10,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))


# ### Grid search

# In[79]:

from sklearn.model_selection import GridSearchCV
# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.5, 0.7],
    'n_estimators': [50],
    'max_depth': [10, 20, 30]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm,param_grid=gbm_param_grid,scoring="neg_mean_squared_error",cv=4)

# Fit grid_mse to the data
grid_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


# ### Randomized search

# In[82]:

from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': range(2,12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform randomized search: randomized_mse
randomized_mse = RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid,scoring="neg_mean_squared_error",cv=4,n_iter=20)

# Fit grid_mse to the data
randomized_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


# In[ ]:



