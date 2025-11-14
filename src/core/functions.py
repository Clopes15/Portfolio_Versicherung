import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statsmodels.formula.api as smf

def heatmap_corr(df):
    correlation = df.dropna().corr()
    sns.set_theme()
    plt.figure(figsize = (22,10))
    sns.heatmap(correlation, annot=True, cbar=True , cmap='RdBu', center = 0)
    plt.title("Correlation Matrix - Heatmap", fontsize=16)
    plt.tight_layout()


def logistic_reg(predict_val, indep_var, dataframe):
    """
    Logistic regression with probability curve plot.

    Args:
        predict_val (str): name of the column with the variable to be predicted
        indep_var (str): name of the column with the independent variable
        dataframe (DataFrame): DataFrame with the data
        
    Returns:
        prob_X: predicted probabilities for X
    """
    df = dataframe.copy()
    df = df[[predict_val, indep_var]].dropna().reset_index()
    formula = 'Q("{}") ~ Q("{}")'.format(predict_val, indep_var)
    model = smf.logit(formula = formula, data = df) 
    results = model.fit() 
    print(results.summary())
    
    #X = pd.Series(range(df[indep_var].min(), df[indep_var].min(), 100)) # Funcao range só funciona com int
    #X = df[indep_var]
    X = np.linspace(df[indep_var].min(), df[indep_var].max(), 100)
    intercept = results.params['Intercept']
    slope = results.params['Q("{}")'.format(indep_var)]
    prob_X = 1 / (1 + math.e ** -(intercept + slope * X))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dataframe[indep_var], dataframe[predict_val], alpha=0.5, label="Real data")
    ax.plot(X, prob_X, color='green', label="Estimated probability")
    ax.set_title("Logistic Regression")
    ax.set(xlabel = indep_var, ylabel = ('P ({} = 1)'.format(predict_val)))
    x_pos = dataframe[indep_var].max()*0.7
    y_pos = 0.75
    ax.legend(loc='center left', bbox_to_anchor=(x_pos, y_pos), frameon=True, borderaxespad=0, bbox_transform=ax.transData)
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
    prob_thr = 0.5
    idx = np.argmax(prob_X >= prob_thr)  
    x_value = X[idx]
    ax.vlines(x=x_value, ymin=0, ymax=1, colors='red', label='Threshold 0.5')
    ann_ax = ax.annotate(xycoords = 'axes fraction',
                         text = 'For customers with a normalized weighted\nrisk factor higher than {:.3f}, there is a {:.0f}%\nprobability of filing an insurance claim'.
                         format(x_value, prob_thr*100), xytext = [x_value+0.1, 0.442], xy = [x_value+0.02, 0.5], arrowprops=dict(facecolor='black'), fontsize = 12)
    print(x_value)
    return prob_X
