import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def show_histogram(data):
    plt.figure(figsize=(8, 6))
    plt.hist(data['Price'], bins=50, edgecolor='black')
    plt.title('Histogram of Price')
    plt.xlabel('Price (in NIS)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(data['NumStores'], data['DogParkInd'], alpha=0.5)
    plt.title('NumStores vs DogParkInd')
    plt.xlabel('Number of Stores')
    plt.ylabel('Dog Park (1 = Yes, 0 = No)')
    plt.grid(True)
    plt.show()


def estimate_model(data):
    model1_predictors = sm.add_constant(data[['MtrsToBeach', 'SqMtrs', 'Age', 'NumStores', 'DogParkInd', 'SchoolScores']])
    y = data['Price']
    model1 = sm.OLS(y, model1_predictors).fit()

    print("Model 1 Summary:")
    print(model1.summary())

    X2 = sm.add_constant(data[['MtrsToBeach', 'SqMtrs', 'Age']])
    model2 = sm.OLS(y, X2).fit()

    print("Model 2 Summary:")
    print(model2.summary())
    return model1, model2


def price_difference(model1, model2):
    sq_mtrs_coef_model1 = model1.params['SqMtrs']
    price_difference_model1 = sq_mtrs_coef_model1 * 18
    print(f"Price difference in Model 1 for 18 m² change: {price_difference_model1} NIS")

    sq_mtrs_coef_model2 = model2.params['SqMtrs']
    price_difference_model2 = sq_mtrs_coef_model2 * 18
    print(f"Price difference in Model 2 for 18 m² change: {price_difference_model2} NIS")


def significant_check(model1, model2):
    f_stat_p_value_model1 = model1.f_pvalue
    print(f"F-statistic p-value for Model 1: {f_stat_p_value_model1}")

    f_stat_p_value_model2 = model2.f_pvalue
    print(f"F-statistic p-value for Model 2: {f_stat_p_value_model2}")


def pvalues_check(model1, model2):
    print("P-values for Model 1:")
    print(model1.pvalues)

    print("P-values for Model 2:")
    print(model2.pvalues)


def vars_impact(data):
    model1_predictors_without_NumStores = data[['MtrsToBeach', 'SqMtrs', 'Age', 'DogParkInd', 'SchoolScores']]
    model1_predictors_without_NumStores = sm.add_constant(model1_predictors_without_NumStores)

    y = data['Price']
    model1_without_NumStores = sm.OLS(y, model1_predictors_without_NumStores).fit()

    print(f"DogParkInd coefficient without NumStores: {model1_without_NumStores.params['DogParkInd']}")


def exercise_1():
    file_path = 'HousePricesHW1.csv'
    data = pd.read_csv(file_path)

    print(data.info())
    print(data.head())

    show_histogram(data=data)

    model1, model2 = estimate_model(data=data)

    price_difference(model1=model1, model2=model2)

    significant_check(model1=model1, model2=model2)

    pvalues_check(model1=model1, model2=model2)

    vars_impact(data=data)


if __name__ == '__main__':
    exercise_1()
