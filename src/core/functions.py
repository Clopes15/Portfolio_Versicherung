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

    #X = pd.Series(range(df.shape[0]))
    #X = df[indep_var]
    X = np.linspace(-0.1, 1.5, 100)
    intercept = results.params['Intercept']
    slope = results.params['Q("{}")'.format(indep_var)]
    prob_X = 1 / (1 + math.e ** -(intercept + slope * X))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dataframe[indep_var], dataframe[predict_val], alpha=0.5, label="Real data")
    ax.plot(X, prob_X, color='green', label="Estimated probability")
    ax.set_title("Logistic Regression")
    ax.set(xlabel = indep_var, ylabel = ('P ({} = 1)'.format(predict_val)))
    x_pos = dataframe[indep_var].max()
    y_pos = 0.5
    ax.legend(loc='center left', bbox_to_anchor=(x_pos, y_pos), frameon=True, borderaxespad=0, bbox_transform=ax.transData)
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
    return prob_X
