import pandas as pd
import utils as utils

def mean_squared_error(X, y, theta0, theta1):
    m = len(X)
    sum_squared_errors = 0
    for i in range(m):
        prediction = utils.estimate_price(X[i], theta0, theta1)
        sum_squared_errors += (prediction - y[i]) ** 2
    return sum_squared_errors / m

def gradient_descend(X, y, m, theta0 = 0, theta1 = 0, alpha = 0.1, iterations = 1000):
    for _ in range(iterations):
        sum_error0 = 0
        sum_error1 = 0
        for i in range(m):
            error = utils.estimate_price(X[i], theta0, theta1) - y[i]
            sum_error0 += error
            sum_error1 += error * X[i]
        
        tmp_theta0 = theta0 - (alpha * sum_error0 / m)
        tmp_theta1 = theta1 - (alpha * sum_error1 / m)

        theta0 = tmp_theta0
        theta1 = tmp_theta1

    return theta0, theta1

def main():
    data = pd.read_csv('data.csv')

    X = data['km'].values
    y = data['price'].values

    # Normalize the data
    X, X_mean, X_std = utils.normalize_data(X)
    y, y_mean, y_std = utils.normalize_data(y)

    theta0, theta1 = gradient_descend(X, y, len(X))

    # De-standardize the parameters
    theta0, theta1 = utils.denormalize_parameters(theta0, theta1, X_mean, X_std, y_mean, y_std)

    print(f"Trained model parameters: theta0={theta0}, theta1={theta1}")

if __name__ == "__main__":
    main()