import os
import pandas as pd


def mean_absolute_error(X, true_values, theta0, theta1):
    m = len(X)
    total_error = 0.0

    for i in range(m):
        predicted_value = theta0 + theta1 * X[i]
        total_error += abs(predicted_value - true_values[i])

    return total_error / m


def main():
    data_file = 'data.csv'
    if not os.path.exists(data_file):
        print(f"Error: The file '{data_file}' does not exist.")
        return
    
    theta_file = 'model_parameters.csv'
    if not os.path.exists(theta_file):
        print(f"Error: The file '{theta_file}' does not exist.")
        return

    data = pd.read_csv('data.csv')
    X = data['km'].values
    true_values = data['price'].values

    thetas = pd.read_csv('model_parameters.csv', header=None).values.flatten()
    theta0 = thetas[0]
    theta1 = thetas[1]

    mae = mean_absolute_error(X, true_values, theta0, theta1)
    print(f'Mean Absolute Error (MAE): {mae:.2f}€')


if __name__ == '__main__':
    main()
