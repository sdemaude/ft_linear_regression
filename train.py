import pandas as pd

def mean_absolute_error(X, y, theta0, theta1):
    predictions = theta0 + theta1 * X
    errors = predictions - y
    return sum(abs(errors)) / len(X)

def gradient_descend(X, y, theta0, theta1, alpha, iterations):
    for _ in range(iterations):   
        if mean_absolute_error(X, y, theta0, theta1) > 0.4:
            break
        predictions = theta0 + theta1 * X
        errors = predictions - y
        
        theta0 -= alpha * (1 / len(X)) * sum(errors)
        theta1 -= alpha * (1 / len(X)) * sum(errors * X)

    return theta0, theta1

def main():
    # Load the dataset
    data = pd.read_csv('data.csv')

    # Extract features and target variable
    X = data['km'].values
    y = data['price'].values

    # Normalize the features
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    # Initialize parameters
    theta0 = 0
    theta1 = 0
    alpha = 0.01  # Learning rate
    iterations = 1000

    theta0, theta1 = gradient_descend(X, y, theta0, theta1, alpha, iterations)

    print(f"Trained model parameters: theta0={theta0}, theta1={theta1}")

if __name__ == "__main__":
    main()