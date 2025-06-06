import numpy as np


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def normalize_data(data):
    mean = sum(data) / len(data)
    
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    
    if std == 0:
        std = 1
    
    normalized_data = [(x - mean) / std for x in data]
    normalized_data = np.array(normalized_data)
    
    return normalized_data, mean, std


def denormalize_parameters(theta0, theta1, mileages_mean, mileages_std, prices_mean, prices_std):
    
    original_theta1 = theta1 * (prices_std / mileages_std)
    original_theta0 = prices_mean + prices_std * theta0 - original_theta1 * mileages_mean

    return original_theta0, original_theta1