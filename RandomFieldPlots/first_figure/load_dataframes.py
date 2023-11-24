import pandas as pd
import numpy as np
import os

# List of test names for reference
#test_names = ['test2_mat2', 'test1_mat2', 'test6_mat1']

# Globally visible list to store crypt dataframes
#crypt_data_list = []

def load_data(category, crypt_number, test_number, mat_num):
    file_name = f'crypt{crypt_number}/{category}_RF-crypt{crypt_number}_test{test_number}_mat{mat_num}.npy'
    file_path = os.path.join('.', file_name)
    matrix_data = np.load(file_path)

    # Check if matrix_data is 1-dimensional
    if matrix_data.ndim == 1:
        matrix_data = matrix_data.reshape(-1, 1)

    matrix_df = pd.DataFrame(matrix_data, columns=[file_name])
    return matrix_df

def main(crypt_data_list, test_numbers = ['2','1','6'], 
         mat_numbers=['2','2', '1'], 
         test_names = ['test2_mat2', 'test1_mat2', 'test6_mat1']):
    # Crypt numbers and test numbers for each case
    crypt_numbers = ['0', '1', '2']
    #     test_numbers = ['2', '1', '6']
    #     mat_numbers = ['2', '2', '1']  # Load only mat2 for now
    

    #global crypt_data_list  # Use the globally visible list

    for i in range(len(crypt_numbers)):
        crypt_data_sublist = []  # List for each crypt
        for category in ['coarse', 'smooth', 'para']:
            data = load_data(category, crypt_numbers[i], test_numbers[i], mat_numbers[i])
            crypt_data_sublist.append(data)
            print(f"\nCRYPT{crypt_numbers[i]} - {category} Data for {test_names[i]}:")
            print(data.head())

        # Append the sublist for the current crypt to the global list
        crypt_data_list.append(crypt_data_sublist)

    return crypt_data_list
    #Example: Accessing the data for crypt 0, test 2, mat 2, COARSE
    #print("\nExample: Accessing data for CRYPT0, TEST2_MAT2, COARSE:")
    #print(crypt_data_list[0][0][0].head())


if __name__ == "__main__":
    main()