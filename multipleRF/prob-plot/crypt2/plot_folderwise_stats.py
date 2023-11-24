import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family": "serif",
})


def load_npy_files(folder_path):
	df = pd.DataFrame({})
	filenames = []
	for file_name in os.listdir(folder_path):
		if file_name.endswith(".npy"):
			file_path = os.path.join(folder_path, file_name)
			fname = os.path.splitext(file_name)[0]
			loaded_data = np.load(file_path)
			loaded_data = np.clip(loaded_data, 0, 1)
			df[fname] = loaded_data
			filenames.append(fname)
	return df, filenames

def save_figure(df, foldername, filenames):
	# Plot the first subplot
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(df['median'], 'b')
	plt.fill_between(df.index, df['median'] - df['mad'], df['median'] + df['mad'], color='blue', alpha=0.2, label='Mean ± Std Dev')
	plt.title("Median Probability Over Time")
	plt.xlabel("Time")
	plt.ylabel("Probability")
	plt.ylim(-0.05, 1.05)

	# Plot the second subplot
	plt.subplot(1, 2, 2)
	plt.plot(df['rate_of_change_of_median'], 'r')
	plt.title("Rate of Change of Median Probabilities over Time")
	plt.xlabel("Time")
	plt.ylabel("Rate of Change")
	plt.ylim(0, 0.05)
	plt.tight_layout()
	plt.savefig(f"{foldername}_median+roc_median.png", dpi=600, bbox_inches="tight")
	plt.clf()

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(df['mean'], label='Mean', color="blue")
	plt.fill_between(df.index, df['mean'] - df['std_dev'], df['mean'] + df['std_dev'], color='blue', alpha=0.2, label='Mean ± Std Dev')
	plt.title("Mean Probability Over Time")
	plt.xlabel("Time")
	plt.ylabel("Probability")
	plt.ylim(-0.05, 1.05)
	plt.tight_layout()


	plt.subplot(1, 2, 2)
	plt.plot(df['rate_of_change_of_mean'], 'r')
	plt.title("Rate of Change of Mean Probability over Time")
	plt.xlabel("Time")
	plt.ylabel("Rate of Change")
	plt.ylim(0, 0.05)
	plt.tight_layout()
	plt.savefig(f"{foldername}_rate_of_change_mean+mean.png", dpi=600, bbox_inches="tight")

	base_name = "Random Field"
	new_names = [f"{base_name} {i + 1}" for i in range(len(filenames))]
	#print(new_names)
	#print(df)
	new_df = df[filenames].copy()
	new_df.columns = new_names
	new_df['Mean'] = df['mean']

	plt.figure(figsize=(10, 6))
	sns.violinplot(data=new_df, palette='Set2')
	plt.title('Distribution of Probabilities for different Realizations of Random Field')
	plt.xticks(rotation=15)  # Change the angle as needed
	# Show the plot
	plt.tight_layout()
	plt.savefig(f"{foldername}_vilion+mean.png", dpi=600, bbox_inches="tight")
	plt.clf()

	sns.kdeplot(data=new_df, fill=True)
	plt.title("Kernel Density Estimation")
	plt.tight_layout()
	plt.savefig(f"{foldername}_kde_plot+mean.png", dpi=600, bbox_inches="tight")
	plt.clf()

	column_to_delete = 'Mean'  # Replace 'Column_Name' with the actual name of the column
	new_df.drop(column_to_delete, axis=1, inplace=True)

	new_df['Median'] = df['median']

	plt.figure(figsize=(10, 6))
	sns.violinplot(data=new_df, palette='Set2')
	plt.title('Distribution of Probabilities for different Realizations of Random Field')
	plt.xticks(rotation=15)  # Change the angle as needed
	# Show the plot
	plt.tight_layout()
	plt.savefig(f"{foldername}_vilion+median.png", dpi=600, bbox_inches="tight")
	plt.clf()

	sns.kdeplot(data=new_df, fill=True)
	plt.title("Kernel Density Estimation")
	plt.tight_layout()
	plt.savefig(f"{foldername}_kde_plot+median.png", dpi=600, bbox_inches="tight")
	plt.clf()

	column_to_delete = 'Median'  # Replace 'Column_Name' with the actual name of the column
	new_df.drop(column_to_delete, axis=1, inplace=True)



def process_all_folders():
	current_directory = os.getcwd()
	for folder_name in os.listdir(current_directory):
		folder_path = os.path.join(current_directory, folder_name)
		if os.path.isdir(folder_path):
			print(f"Processing folder: {folder_path}")
			process_folder(folder_path)

def process_folder(folder_path):
	df, filenames = load_npy_files(folder_path)
	df = df.applymap(lambda x: 0 if x < 0 else (1 if x > 1 else x))
	max_values = df.max()
	min_values = df.min()

	# Print the results
	print("Maximum values for each column:")
	print(max_values)

	print("\nMinimum values for each column:")
	print(min_values)

	df['std_dev'] = df.std(axis=1)
	df['median'] = df.median(axis=1)
	plot_df = df.melt(id_vars='median', var_name='Column', value_name='Value')
	df['mean'] = df.mean(axis=1)
	mad_values = np.median(np.abs(df.sub(df['median'], axis=0)), axis=1)
	df['mad'] = mad_values
	df['rate_of_change_of_mean'] = np.gradient(df['mean'])
	df['rate_of_change_of_median'] = np.gradient(df['median'])

	save_figure(df, folder_path, filenames)

def main():
	process_all_folders()

if __name__ == "__main__":
	main()

