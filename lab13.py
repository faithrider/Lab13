"""
Regresssion Lab

The program both prints to the screen, and writes to another file. The results are the same.


"""
# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import fetch_california_housing


# open txt file (to write table to)
table = open("cali_housing_table.txt", "w")

# read california housing data
cali = fetch_california_housing()
cali_df = pd.DataFrame(cali.data,columns=cali.feature_names)
cali_df['MedHouseValue'] = pd.Series(cali.target)


# performs the simple linear regression
def do_simple_regression(i):
    # set variables
    y = pd.DataFrame(cali_df["MedHouseValue"])
    X = pd.DataFrame(cali_df[cali.feature_names[i]])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    # perform the regression
    regression = LinearRegression()
    regression.fit(X_train, y_train)
    y_pred = regression.predict(X_test)

    # get results
    R2 = metrics.r2_score(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)

    # print results
    print(f"Feature %d has R2 score  : %f" %(i, R2))
    print(f"\t  has MSE score : %f\n" %(MSE))

    # write results
    table.write(f"Feature %d has R2 score  : %f\n" %(i, R2))
    table.write(f"\t  has MSE score : %f\n" %(MSE))



# performs the multiple linear regression
def do_multiple_regression():
    # set variables
    y = pd.DataFrame(cali_df["MedHouseValue"])
    X = pd.DataFrame(cali_df.loc[:, 'MedInc':'Longitude'])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    # perform the regression
    regression = LinearRegression()
    regression.fit(X_train, y_train)
    y_pred = regression.predict(X_test)

    # get results
    R2 = metrics.r2_score(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    
    # print results
    print("Multiple Linear Regression using All features")
    print(f"R2 score : %f" %(R2))
    print(f"MSE score: %f\n" %(MSE))

    # write results
    table.write("Multiple Linear Regression using All features\n")
    table.write(f"R2 score : %f\n" %(R2))
    table.write(f"MSE score: %f\n\n" %(MSE))




# do the multiple linear regression
do_multiple_regression()

# do the simple linear regressions
for i in range(len(cali.feature_names)):
    do_simple_regression(i)