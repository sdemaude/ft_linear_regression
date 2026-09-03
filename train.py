import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train a linear regression model.')
    parser.add_argument('-v', '--visualize', action='store_true', help='show real-time training visualization')
    return parser.parse_args()


def mean_squared_error(X, y, m, theta0, theta1):
    total_error = 0
    for i in range(m):
        prediction = utils.estimate_price(X[i], theta0, theta1)
        total_error += (prediction - y[i]) ** 2
    return total_error / m


def init_plot(X, y):
    plt.ion() # Enable interactive mode for real-time updates
    
    _, ax = plt.subplots()

    ax.scatter(X, y, color='LightCoral', label='Data')
    regression_line, = ax.plot([], [], color='OliveDrab', label='Regression Line') # Empty line to be updated later

    ax.set_xlabel('Mileage (normalized)')
    ax.set_ylabel('Price (normalized)')
    ax.legend()
    return regression_line


def update_plot(line, X, theta0, theta1):
    x_min, x_max = min(X), max(X)
    y_min = utils.estimate_price(x_min, theta0, theta1)
    y_max = utils.estimate_price(x_max, theta0, theta1)

    line.set_data([x_min, x_max], [y_min, y_max])
    plt.draw()
    plt.pause(0.01)
    time.sleep(0.1)


def linear_regression(X, y, alpha=0.1, iterations=150, visualize=False):
    theta0 = 0
    theta1 = 0
    m = len(X)

    if visualize:
        alpha = 0.05
        iterations = 300
        line = init_plot(X, y)

    for epoch in range(iterations):
        mse = mean_squared_error(X, y, m, theta0, theta1)
        if mse < 0.1:
            print(f'Converged at iteration {epoch} with MSE={mse:.6f}')
            break

        if visualize:
            print(f'Iteration {epoch}: MSE={mse:.6f}, theta0 (normalized) = {theta0:.6f}, theta1 (normalized) = {theta1:.6f}')

        sum_error_theta0 = 0
        sum_error_theta1 = 0

        for i in range(m):
            error = utils.estimate_price(X[i], theta0, theta1) - y[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * X[i]

        theta0 -= alpha * sum_error_theta0 / m
        theta1 -= alpha * sum_error_theta1 / m

        if visualize:
            update_plot(line, X, theta0, theta1)

    if visualize:
        plt.ioff()
        plt.show()

    return theta0, theta1


def main():
    args = parse_args()

    data_file = 'data.csv'
    if not os.path.exists(data_file):
        print(f"Error: The file '{data_file}' does not exist.")
        return

    data = pd.read_csv(data_file)
    
    required_columns = ['km', 'price']
    if not all(col in data.columns for col in required_columns):
        print(f"Missing required columns in the dataset. Expected columns: {required_columns}")
        return

    X = data['km'].values
    y = data['price'].values

    X, X_mean, X_std = utils.normalize_data(X)
    y, y_mean, y_std = utils.normalize_data(y)

    theta0, theta1 = linear_regression(X, y, visualize=args.visualize)

    theta0, theta1 = utils.denormalize_parameters(theta0, theta1, X_mean, X_std, y_mean, y_std)

    print(f'Trained model parameters: theta0 = {theta0:.6f}, theta1 = {theta1:.6f}')

    with open('model_parameters.csv', 'w') as f:
        f.write(f"{theta0},{theta1}")


if __name__ == '__main__':
    main()
