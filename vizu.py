import pandas as pd
import matplotlib.pyplot as plt

def plot_data(df):
    plt.scatter(df['km'], df['price'], color='blue', alpha=0.5)
    plt.title('Scatter plot of km vs price')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.grid()
    plt.show()

def main():
    df = pd.read_csv('data.csv')

    if 'km' not in df.columns or 'price' not in df.columns:
        print("Error: The CSV file must contain 'km' and 'price' columns.")
        return

    plot_data(df)

if __name__ == "__main__":
    main()