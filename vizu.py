import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(df, param):
    x = df['km']
    y = df['price']

    plt.scatter(x, y, color='LightCoral', label='Data', alpha=0.5)

    if not param.empty:
        b = param.iloc[0, 0]
        a = param.iloc[0, 1]

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = a * x_line + b
        plt.plot(x_line, y_line, color='OliveDrab', label=f'Regression: y = {a:.2f}x + {b:.2f}')

    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Price vs Mileage with Regression Line')
    plt.grid()
    plt.show()


def main():
    data_file = 'data.csv'
    if not os.path.exists(data_file):
        print(f"Error: The file '{data_file}' does not exist.")
        return
    df = pd.read_csv('data.csv')
    param = pd.DataFrame()
    
    try:
        param = pd.read_csv('model_parameters.csv', header=None)
    except FileNotFoundError:
        print("Parameter file not found. Regression line will not be shown.")

    plot_data(df, param)

if __name__ == "__main__":
    main()