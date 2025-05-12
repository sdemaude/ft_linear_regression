import utils as utils

def main():
    theta0 = 3.8117657178797034e-17
    theta1 = -0.1416792924209209

    try:
        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = utils.estimate_price(mileage, theta0, theta1)
        print(f"Estimated price for a vehicule with a mileage of {mileage} km is: {estimated_price}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")

if __name__ == "__main__":
    main()