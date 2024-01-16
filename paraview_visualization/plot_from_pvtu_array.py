import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_lines(file_list, x_label_ = "", y_label_ ="", title="", savefig_name="default.pdf"):
    for array in file_list:
        plt.plot(array, '--')
    plt.xlabel(x_label_)
    plt.ylabel(y_label_)
    plt.title(title)
    plt.savefig(savefig_name)


name = "crypt1_overall"
OVERALL = [np.load(f"crypt1_test{i}_overall.npy") for i in range(6)]

plot_lines(OVERALL, savefig_name = name)


d = {f'col{i}': array for i, array in enumerate(OVERALL)}
df = pd.DataFrame(data=d)

# row_medians = df.median(axis=1)
row_stats = df.agg(['min', 'max', 'median'], axis=1)
# plt.plot(row_stats['median'])
# plt.plot(row_stats['min'])
# plt.plot(row_stats['max'])

medians = np.median(df, axis=1)
mins = np.min(df, axis=1)
maxs = np.max(df, axis=1)

plt.figure(figsize=(8, 6))
sns.lineplot(x=range(len(medians)), y=medians, color='indigo')
plt.fill_between(range(len(mins)), mins, maxs, color='indigo', alpha=0.1)
# plt.xlabel('Rows')
# plt.ylabel('Values')
plt.legend()
plt.show()
