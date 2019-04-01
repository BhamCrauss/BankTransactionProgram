import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
from datetime import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math


######### graph setup ##########################

fig2 = plt.figure()
fig2.canvas.set_window_title('Dingout vs Groceries')
rect2 = fig2.patch
rect2.set_facecolor('#31312e')

fig = plt.figure()
fig.canvas.set_window_title('Monthly Breakdown')
rect = fig.patch
rect.set_facecolor('#31312e')

###############  DATA ENTRY STUFF   ###############################################

#info= input("Enter your banking transaction file in csv format:")
#file = pd.read_csv(info)
file = pd.read_csv('/Users/student/Desktop/pythonStuff/03:11:18-3:11:19.csv')


file.index.name = 'index'
file['created_at'] = pd.to_datetime(file['created_at'])
datetime_object = datetime.strptime('12/29/2018 23:04', '%m/%d/%Y %H:%M')

data = file[['created_at','nickname','tags', 'amount']]
data = data.copy()

year = data.created_at.dt.year
data['year'] = year
month = data.created_at.dt.month  
data['month'] =  month

##week = data.created_at.dt.month  
##data['week'] =  week 

data2 = data[data['nickname'] != 'DEPOSIT']
data2 = data2[data2['tags'] != 'Income']
data2 = data2[data2['tags'] != 'Payment']
data2 = data2[data2['tags'] != 'Fees']
data2 = data2[data2['tags'] != 'Utilities']
data2 = data2[data2['tags'] != 'Health']
data2 = data2[data2['tags'] != 'Education']
data2 = data2[data2['amount'] < 1100]


##### to isolate dining out and groceries ########3

data3 = data2[data2['tags'] != 'Entertainment']
data3 = data3[data3['tags'] != 'Home']
data3 = data3[data3['tags'] != 'Personal']
data3 = data3[data3['tags'] != 'Transportation']


######### gas ######################
gas = (data2[data2['tags']=='Transportation'])
gas = (gas[gas['amount'] > 8])
gas = (gas[gas['amount'] < 50])

####################  DATA SORTING ################################

#data3 = data2[data2['amount'] < 1100]
#print(data2)
#print(data3)

data5 = pd.pivot_table(data3,index=["year", "month"], columns=["tags"], aggfunc='sum')
data5.fillna(0, inplace=True)

##############  Averages By Catagory ################################

sumdf = pd.pivot_table(data,index=["year", "month"], columns=["tags"], aggfunc='sum')

entertainmentSum = (sumdf['amount']['Entertainment'])
healthSum = (sumdf['amount']['Health'])
homeSum = (sumdf['amount']['Home'])
utilitiesSum = (sumdf['amount']['Utilities'])

dineOutSum = (data5['amount']['Diningout'])
groceriesSum = (data5['amount']['Groceries'])

print(sumdf)

print('Dining Out Monthly Average - $',round(dineOutSum.mean(), 2))
print('Grocery Monthly Average - $',round(groceriesSum.mean(), 2))
print('Entertainment Monthly Average - $',round(entertainmentSum.mean(), 2))
print('Health Monthly Average - $',round(healthSum.mean(), 2))
print('Home Monthly Average - $',round(homeSum.mean(), 2))
print('Utilities Monthly Average - $',round(utilitiesSum.mean(), 2))


data6 = pd.pivot_table(data2,index=["year", "month"], columns=["tags"], aggfunc='sum')
data6.fillna(0, inplace=True)


####### monthly breakdown Graph #############################
ax1 = fig.add_subplot(2,1,2, facecolor='grey'
                      )
data6.plot.bar(ax=ax1)
ax1.tick_params(axis='x', colors='c')
ax1.tick_params(axis='y', colors='c')
ax1.spines['bottom'].set_color('w')
ax1.spines['top'].set_color('w')
ax1.spines['left'].set_color('w')
ax1.spines['right'].set_color('w')
ax1.yaxis.label.set_color('c')
ax1.xaxis.label.set_color('c')
ax1.set_title('Monthly Totals by Catagory', color = 'c')
#ax1.set_xlabel('Dollars$')
ax1.set_ylabel('Monthly Sum $')
ax1.legend(loc=0, fontsize = 'x-small')

########## dineout vs groceries graph ####################

ax2 = fig2.add_subplot(2,2,1, facecolor='grey')
data5.plot(ax=ax2, linewidth=3.3)
ax2.tick_params(axis='x', colors='c')
ax2.tick_params(axis='y', colors='c')
ax2.spines['bottom'].set_color('w')
ax2.spines['top'].set_color('w')
ax2.spines['left'].set_color('w')
ax2.spines['right'].set_color('w')
ax2.yaxis.label.set_color('c')
ax2.xaxis.label.set_color('c')
ax2.set_title('DiningOut vs Groceries', color = 'c')
ax2.set_xlabel('Month')
ax2.set_ylabel('Amount Spent $')


###################   new graphs (income, gas, frivolous)   #################################

##### INCOME GRAPH ############
income1 = data[data['nickname'] == 'DEPOSIT']
income1 = data[data['tags'] == 'Income']

income1 = income1[income1['nickname'] != 'APPLIANCEPARTSP']
income1 = income1[income1['nickname'] != 'SOUTHEASTERN C']
income1 = income1[income1['nickname'] != 'REDEMPTIVE CYC']
income1 = income1[income1['nickname'] != 'LAWSON STATE CO']
income1 = income1[income1['nickname'] != 'Interest Payment']
income1 = income1[income1['nickname'] != 'ELIQUIDDEPOT CO']


income2 = pd.pivot_table(income1,index=["year", "month"], columns=["nickname"], aggfunc='sum')
income2.fillna(0, inplace=True)

ax4 = fig.add_subplot(2,3,1, facecolor='grey')
income2.plot(ax=ax4, linewidth=3.3)
ax4.tick_params(axis='x', colors='c')
ax4.tick_params(axis='y', colors='c')
ax4.spines['bottom'].set_color('w')
ax4.spines['top'].set_color('w')
ax4.spines['left'].set_color('w')
ax4.spines['right'].set_color('w')
ax4.yaxis.label.set_color('c')
ax4.xaxis.label.set_color('c')
ax4.set_title('Income', color = 'c')
ax4.set_xlabel('Month')
ax4.set_ylabel('Amount Spent $')
ax4.legend(loc=1, fontsize = 'xx-small')

##### gas graph ############
gas2 = pd.pivot_table(gas,index=["year", "month"], aggfunc='sum')
gas2.fillna(0, inplace=True)


ax5 = fig.add_subplot(2,3,2, facecolor='grey')
gas2.plot.bar(ax=ax5)
ax5.tick_params(axis='x', colors='c')
ax5.tick_params(axis='y', colors='c')
ax5.spines['bottom'].set_color('w')
ax5.spines['top'].set_color('w')
ax5.spines['left'].set_color('w')
ax5.spines['right'].set_color('w')
ax5.yaxis.label.set_color('c')
ax5.xaxis.label.set_color('c')
ax5.set_title('Gas Purchases', color = 'c')
#ax5.set_xlabel('Dollars$')
ax5.set_ylabel('Monthly Sum $')
ax5.legend(loc=0, fontsize = 'x-small')


##### frivolous ############

friv1 = data[data['tags'] != 'Income']
friv2 = friv1[friv1['amount'] < 7]

friv3 = pd.pivot_table(friv2,index=["year", "month"], aggfunc='sum')
friv3.fillna(0, inplace=True)


ax5 = fig.add_subplot(2,3,3, facecolor='grey')
friv3.plot.bar(ax=ax5)
ax5.tick_params(axis='x', colors='c')
ax5.tick_params(axis='y', colors='c')
ax5.spines['bottom'].set_color('w')
ax5.spines['top'].set_color('w')
ax5.spines['left'].set_color('w')
ax5.spines['right'].set_color('w')
ax5.yaxis.label.set_color('c')
ax5.xaxis.label.set_color('c')
ax5.set_title('Frivoulous Spending', color = 'c')
#ax5.set_xlabel('Dollars$')
ax5.set_ylabel('Monthly Sum $')
ax5.legend(loc=0, fontsize = 'x-small')

###############  REGRESSION STUFF   ############################################



### DataFrame #####

df = pd.DataFrame()
df['dining_out'] = dineOutSum
df['groceries'] = groceriesSum
df['food_sum'] = df.sum(axis=1)

#print(df)

#### split test stuff ########

x_train, x_test, y_train, y_test = train_test_split(df[['dining_out']],df[['groceries']],test_size=0.2)

clf = linear_model.LinearRegression()
clf.fit(x_train,y_train)

#y_prdt = clf.predict(x_test)

##print("split_prediction of **",x_test)
##print("actual values =",y_test)
##print("prediction =",clf.predict(x_test),"***")
##print('r2 score of dine out vs groc=', clf.score(x_test,y_test))
#print('mean squared', mean_squared_error(y_test,y_prdt)) 

########

x2_train, x2_test, y2_train, y2_test = train_test_split(df[['dining_out']],df[['food_sum']],test_size=0.2)

clf2 = linear_model.LinearRegression()
clf2.fit(x2_train,y2_train)

y2_prdt = clf2.predict(x2_test)

##print("split_prediction of **",x2_test)
##print("actual values =",y2_test)
##print("prediction =",clf2.predict(x2_test),"***")
##print('r2 score of dine out vs food sum=', clf2.score(x2_test,y2_test))
##print('mean squared2', mean_squared_error(y2_test,y2_prdt))
##

### Regression of dining out and groceries ######

reg = linear_model.LinearRegression()
reg.fit(df[['dining_out']],df[['groceries']])


##print("dine out vs groc, predict 150",reg.predict([[150]]))
##
##print('coeff =', reg.coef_)
##print('intercept =', reg.intercept_)

ax3 = fig2.add_subplot(2,2,3, facecolor='grey')
ax3.scatter(df.dining_out, df.groceries, color='red')
ax3.plot(df.dining_out,reg.predict(df[['dining_out']]), color='blue')
ax3.tick_params(axis='x', colors='c')
ax3.tick_params(axis='y', colors='c')
ax3.spines['bottom'].set_color('w')
ax3.spines['top'].set_color('w')
ax3.spines['left'].set_color('w')
ax3.spines['right'].set_color('w')
ax3.yaxis.label.set_color('c')
ax3.xaxis.label.set_color('c')
ax3.set_title('Dining Out vs Grocery', color = 'c')
ax3.set_xlabel('Dining-Out Per Month $')
ax3.set_ylabel('Groceries Per Month $')

##########   regression of groceries vs total food cost   ############################
reg2 = linear_model.LinearRegression()
reg2.fit(df[['groceries']],df[['food_sum']])


##print(reg2.predict([[150]]))
##
##print('coeff2 =', reg2.coef_)
##print('intercept2 =', reg2.intercept_)

ax3 = fig2.add_subplot(2,2,4, facecolor='grey')
ax3.scatter(df.groceries, df.food_sum, color='red')
ax3.plot(df.groceries,reg2.predict(df[['groceries']]), color='blue')
ax3.tick_params(axis='x', colors='c')
ax3.tick_params(axis='y', colors='c')
ax3.spines['bottom'].set_color('w')
ax3.spines['top'].set_color('w')
ax3.spines['left'].set_color('w')
ax3.spines['right'].set_color('w')
ax3.yaxis.label.set_color('c')
ax3.xaxis.label.set_color('c')
ax3.set_title('Grocery vs Food Total Correlation', color = 'c')
ax3.set_xlabel('Groceries Per Month $')
ax3.set_ylabel('Total Food $')

###############     regression of dining-out vs total food cost    ##########################

reg3 = linear_model.LinearRegression()
reg3.fit(df[['dining_out']],df[['food_sum']])


##print(reg3.predict([[150]]))
##
##print('coeff3 =', reg3.coef_)
##print('intercept3 =', reg3.intercept_)

ax3 = fig2.add_subplot(2,2,2, facecolor='grey')
ax3.scatter(df.dining_out, df.food_sum, color='red')
ax3.plot(df.dining_out,reg3.predict(df[['dining_out']]), color='blue')
ax3.tick_params(axis='x', colors='c')
ax3.tick_params(axis='y', colors='c')
ax3.spines['bottom'].set_color('w')
ax3.spines['top'].set_color('w')
ax3.spines['left'].set_color('w')
ax3.spines['right'].set_color('w')
ax3.yaxis.label.set_color('c')
ax3.xaxis.label.set_color('c')
ax3.set_title('Dining-Out-Correlation', color = 'c')
ax3.set_xlabel('Dining-Out $')
ax3.set_ylabel('Total Food $')

###### SHOW PLOTS ##########

fig.tight_layout()
fig2.tight_layout()
#plt.show()


