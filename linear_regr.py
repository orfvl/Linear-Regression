import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, t as t_dbn

error = 0.127

def compute_std(X, sigma_hat):
    n = X.size
    return [sigma_hat * np.sqrt(1/(n * np.var(X))) , sigma_hat * np.sqrt(1/n + X.mean()**2/np.var(X)/n)]

df = [pd.read_csv("descending.csv"), pd.read_csv("ascending.csv")]
# df = [pd.read_csv("bending.csv")]
# print(df)
regr = [ linear_model.LinearRegression() for _ in range(len(df))]

y = [f['h'] for f in df]
X = [f.drop('h', axis=1) for f in df]
#X_line is for plotting
X_line = [pd.DataFrame({m: np.linspace(min(x[m].values), max(x[m].values), 10) for m in x}) for x in X]

for i in range(len(df)):
    regr[i].fit(X[i], y[i])
y_line_pred = [ r.predict(x) for r,x in zip(regr, X_line)]
y_pred = [ r.predict(x) for r,x in zip(regr, X)]

#sigma hat squared
sigma_hat = [ np.sqrt(mean_squared_error(y_, y_pred_) * len(y_)/(len(y_)-2)) for y_,y_pred_ in zip(y,y_pred)] 

quantile = norm.ppf(0.95)
beta = [[r.coef_, r.intercept_] for r in regr]

delta = [[ i * quantile for i in compute_std(x.values, s)] for x,s in zip(X,sigma_hat)]

for b,d in zip(beta,delta):
    print("w=", b[0], "+-", d[0])
    print("b=", b[1], "+-", d[1]) 

fig, axs = plt.subplots(len(df), 1)
for i in range(len(df)):
    axs[i].plot(X_line[i]['m'], y_line_pred[i], color="blue", linewidth=3, label = "Linear regression")
    axs[i].errorbar(X[i]['m'].values, y[i] , yerr=[error]*len(y[i]), fmt="o", label="Measurement")

    axs[i].legend()
    axs[i].set_ylabel('s in mm')
    axs[i].set_xlabel('m in kg')

axs[0].set_title("elongation decreasing")
#axs.set_title("bending of a bar of rectangular profile")
axs[1].set_title("elongation increasing")
fig.tight_layout()
fig.savefig("elongation")
# fig.savefig("bending")
plt.show()
