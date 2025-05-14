import pandas as pd
import utils as utils

def mean_squared_error(X, y, m, theta0, theta1):
    sum_squared_errors = 0

    for i in range(m):
        prediction = utils.estimate_price(X[i], theta0, theta1)
        sum_squared_errors += (prediction - y[i]) ** 2

    return sum_squared_errors / m

def gradient_descend(X, y, m, theta0 = 0, theta1 = 0, alpha = 0.1, iterations = 1000):
    for _ in range(iterations):

        sum_error0 = 0
        sum_error1 = 0

        # Calculate the mean squared error
        mse = mean_squared_error(X, y, m, theta0, theta1)
        print(f"Iteration {_}: MSE={mse}, theta0={theta0}, theta1={theta1}")
        if mse < 0.1:
            print(f"Converged at iteration {_} with MSE={mse}")
            break

        for i in range(m):
            error = utils.estimate_price(X[i], theta0, theta1) - y[i]
            sum_error0 += error
            sum_error1 += error * X[i]

        print(f"Iteration {_}: error={error}")
        
        theta0 -= (alpha * sum_error0 / m)
        theta1 -= (alpha * sum_error1 / m)

    return theta0, theta1

def main():
    data = pd.read_csv('data.csv')

    X = data['km'].values
    y = data['price'].values

    X, X_mean, X_std = utils.normalize_data(X)
    y, y_mean, y_std = utils.normalize_data(y)

    theta0, theta1 = gradient_descend(X, y, len(X))
    theta0, theta1 = utils.denormalize_parameters(theta0, theta1, X_mean, X_std, y_mean, y_std)

    print(f"Trained model parameters: theta0={theta0}, theta1={theta1}")

    with open('model_parameters.csv', 'w') as f:
        f.write(f"{theta0}, {theta1}")

if __name__ == "__main__":
    main()