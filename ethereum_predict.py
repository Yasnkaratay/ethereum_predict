
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from selenium import webdriver
from selenium.webdriver.common.by import By
import sched, time, datetime
import tkinter as tk
import threading

data = pd.read_csv("ETHUSDT 15M.csv")
print(data.head())


print(data.corr())

sutun_isimler = ["Index","Zaman","Acılıs","Yuksek","Dusuk","Kapanıs","Hacim"]
data.columns = sutun_isimler
print(data.head())


data = data.drop(["Index"],axis=1)
data['Zaman'] = pd.to_numeric(pd.to_datetime(data['Zaman']))

a = data.iloc[152828:,:4]
b = data.iloc[152828:,-1]
x = pd.concat([a,b],axis=1)
y = data.drop(["Zaman","Acılıs","Hacim","Yuksek","Dusuk"],axis=1)
y = y.iloc[152828:,:]
x = x.drop(["Hacim","Yuksek","Dusuk"],axis=1)

X = x.values
Y = y.values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
lin_reg_predict = lin_reg.predict(x_test)


from sklearn.preprocessing import PolynomialFeatures
x_poly = PolynomialFeatures(degree=2)
poly_reg = x_poly.fit_transform(x_train)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(poly_reg,y_train)
lin_reg_poly_predcit = lin_reg_poly.predict(x_poly.fit_transform(x_test))


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x_train,y_train)
rf_predcit = rf.predict(x_test)

from sklearn.metrics import r2_score
print("r2_score--------------------")
print("linear regrassion")
print(r2_score(y_test,lin_reg_predict))
print("polynımal regrassion")
print(r2_score(y_test,lin_reg_poly_predcit))
print("decision tree")
print(r2_score(y_test,dt_predict))
print( "random forest regrassıon ")
print(r2_score(y_test,rf_predcit))
print("svr")
#print(r2_score(y_testsc,svr_predict))

print("p-value-------------------")
print("Linear Regrassion p-Value")
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print("Polyminal Regression p-Values")
model1 = sm.OLS(lin_reg_poly.predict(x_poly.fit_transform(X)),X)
print(model1.fit().summary())

print("DecisionTreeRegression P-Value")
model2 = sm.OLS(dt.predict(X),X)
print(model2.fit().summary())

print("Random Forest p-Values")
model3 = sm.OLS(rf.predict(X),X)
print(model.fit().summary())
def dongu():
    current_time = datetime.datetime.now()
    if current_time.minute in [0,15,30,45]:
        new_time = current_time.replace(minute=current_time.minute, second=0, microsecond=0) + datetime.timedelta(minutes=15)
        numeric_time = int(pd.Timestamp(new_time).value // 10**0)
        numeric_time = np.int64(numeric_time)

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')

        url = 'https://www.binance.us/'

        driver = webdriver.Chrome(options=options)
        driver.get(url)

        element = driver.find_elements(By.XPATH,'//*[@id="__APP"]/div[1]/div[2]/div/div[3]/div/div/div/div/table/tbody/tr[3]/td[2]/div')
        element = element[0].text
        element = element.replace("$","")
        element = element.replace(",", "")
        element = float(element)
        
        linearAnaliz =  lin_reg.predict([[numeric_time,element]])
        polynımalAnaliz = lin_reg_poly.predict(x_poly.fit_transform([[numeric_time,element]]))
        decisiontreeAnaliz = dt.predict([[numeric_time,element]])
        randomforestAnaliz = rf.predict([[numeric_time,element]])
        print(linearAnaliz)
        print("Linear--------")
        print(polynımalAnaliz)
        print("poly----------")
        print(decisiontreeAnaliz)
        print("decision-------")
        print(randomforestAnaliz)
        print("random----------")
        print(element)        
        print("ETH----------")
        driver.quit()
        
        return linearAnaliz, polynımalAnaliz, decisiontreeAnaliz, randomforestAnaliz, element

def update_label():
    current_time = datetime.datetime.now()
    if current_time.minute not in [0, 25, 30, 45]:
        label.config(text="Lütfen bekleyiniz", font=("Helvetica", 50), bg='black', fg='white')
        root.after(60500, update_label)
        root.update_idletasks()
        return

    results = dongu()
    label.config(text="Analiz 1: {}\nAnaliz 2: {}\nAnaliz 3: {}\nAnaliz 4: {}\nETH : {}".format(*results), font=("Helvetica", 50), bg='black', fg='white')
    root.after(60000, update_label)
    root.update_idletasks()

root = tk.Tk()
root.title("Analiz")
root.config(bg='black')

label = tk.Label(root, text="", font=("Helvetica", 50), bg='black', fg='white')
label.pack()

update_button = tk.Button(root, text="Yenile", command=lambda: threading.Thread(target=update_label, daemon=True).start(), bg='black', fg='white')
update_button.pack()

update_label()
root.mainloop()




