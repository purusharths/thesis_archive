import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family": "serif",  # Set the font family to serif
})

filenames = []

def load_npy_files(folder_path):
    df = pd.DataFrame({})
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            fname = os.path.splitext(file_name)[0]
            loaded_data = np.load(file_path)
            loaded_data = np.clip(loaded_data, 0, 1)
            df[fname] = loaded_data
            filenames.append(fname)
    return df

def save_figure(df, foldername):
    plt.plot(df['median'], 'b')
    plt.fill_between(df.index, df['median'] - df['mad'], df['median'] + df['mad'], color='blue', alpha=0.2, label='Mean ± Std Dev')
    plt.title("Median Probability Over Time")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.savefig(f"{foldername}_median.png", dpi=600)
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(df['rate_of_change'], 'r')
    plt.title("Rate of Change of Probability over Time")
    plt.xlabel("Time")
    plt.ylabel("Rate of Change")
    plt.savefig(f"{foldername}_rate_of_change.png", dpi=600)
    plt.clf()

    plt.plot(df['mean'], label='Mean', color="blue")
    plt.fill_between(df.index, df['mean'] - df['std_dev'], df['mean'] + df['std_dev'], color='blue', alpha=0.2, label='Mean ± Std Dev')
    plt.title("Mean Probability Over Time")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.savefig(f"{foldername}_mean.png", dpi=600)
    plt.clf()

    base_name = "Random Field"
    new_names = [f"{base_name} {i + 1}" for i in range(len(filenames))]
    new_df = df[filenames].copy()
    new_df.columns = new_names
    new_df['Mean'] = df['mean']

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=new_df, palette='Set2')
    plt.title('Distribution of Values from Different Realizations of Random Field')
    plt.xticks(rotation=20)  # Change the angle as needed
    # Show the plot
    plt.savefig(f"{foldername}_vilion+mean.png", dpi=600)
    plt.clf()

    sns.kdeplot(data=new_df, fill=True)
    plt.savefig(f"{foldername}_kde_plot+mean.png", dpi=600)
    plt.clf()

    column_to_delete = 'Mean'  # Replace 'Column_Name' with the actual name of the column
    new_df.drop(column_to_delete, axis=1, inplace=True)

    new_df['Median'] = df['median']

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=new_df, palette='Set2')
    plt.title('Distribution of Values from Different Realizations of Random Field')
    plt.xticks(rotation=20)  # Change the angle as needed
    # Show the plot
    plt.savefig(f"{foldername}_vilion+median.png", dpi=600)
    plt.clf()

    sns.kdeplot(data=new_df, fill=True)
    plt.title("Kernel Density Estimation")
    plt.savefig(f"{foldername}_kde_plot+median.png", dpi=600)
    plt.clf()

    column_to_delete = 'Median'  # Replace 'Column_Name' with the actual name of the column
    #new_df.drop(column_to_delete, axis=1, inplace=True)

def main():
    # Check if a folder path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1]

    # Load all npy files in the specified folder
    df = load_npy_files(folder_path)

    # Create a DataFrame with each column corresponding to a loaded npy file
    df['median'] = df.median(axis=1)
    plot_df = df.melt(id_vars='median', var_name='Column', value_name='Value')
    df['mean'] = df.mean(axis=1)
    mad_values = np.median(np.abs(df.sub(df['median'], axis=0)), axis=1)

    # Add a new column 'mad' containing the MAD for each row
    df['mad'] = mad_values
    # Calculate the standard deviation for each row
    df['std_dev'] = df.std(axis=1)

    # Calculate the rate of change for each row
    df['rate_of_change'] = df.pct_change(axis=1).mean(axis=1)

    save_figure(df, folder_path)

if __name__ == "__main__":
    main()
