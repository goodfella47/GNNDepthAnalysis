import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_tensorboard_csvs(folder, title, xlabel, ylabel):
    """Plot the data from multiple tensorboard csv files."""

    csv_folder = folder
    dfs = []
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            df_name = file.split('-')[1]
            df = pd.read_csv(os.path.join(csv_folder, file))
            dfs.append((df_name, df))
            
    dfs.sort(key=lambda x: x[0])
    for df_name, df in dfs:
        plt.plot(df['Step'], df['Value'], label=df_name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()