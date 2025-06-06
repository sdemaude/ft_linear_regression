import utils
import pandas as pd


def main():
    try:
        model_parameters = pd.read_csv('model_parameters.csv', header=None)
        theta0, theta1 = model_parameters.iloc[0]
    except FileNotFoundError:
        print("Model parameters file not found. Please train the model first.")
        return

    try:
        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = utils.estimate_price(mileage, theta0, theta1)
        print(f"Estimated price for a vehicule with a mileage of {mileage} km is: {estimated_price}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")


if __name__ == "__main__":
    main()