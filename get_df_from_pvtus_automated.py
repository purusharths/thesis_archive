import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import scipy.fftpack

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
import seaborn as sns
import sys


def get_val(x, y, points_array, u_array):
    # Convert x, y to the index in points_array
    idx = np.argmin((points_array[:, 0] - x)**2 + (points_array[:, 1] - y)**2)
    return u_array[idx]

def get_loc(x, y, points_array, u_array):
    # Convert x, y to the index in points_array
    idx = np.argmin((points_array[:, 0] - x)**2 + (points_array[:, 1] - y)**2)
    return tuple(points_array[idx][:2])


def get_obs_and_soln(filename):
    # Read the PVTU file
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(filename) #("crypt/fk_model_sol_0000.pvtu")
    reader.Update()

    # Get the output data
    output = reader.GetOutput()

    # Apply vtkCellDataToPointData filter
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputData(output)
    cell_to_point.Update()
    point_data_output = cell_to_point.GetOutput().GetPointData()

    # Search for the 'u' property in point data arrays
    u_array = None
    num_arrays = point_data_output.GetNumberOfArrays()
    for i in range(num_arrays):
        array_name = point_data_output.GetArrayName(i)
        if array_name == 'u':
            u_array = vtk_to_numpy(point_data_output.GetArray(i))
            break

    # Check if 'u' property is found
    if u_array is None:
        print("Property 'u' not found in the dataset.")
        exit()

    # Convert the data to NumPy arrays
    points_array = vtk_to_numpy(output.GetPoints().GetData())

    # Plot the data
    df = pd.DataFrame({'x': points_array[:, 0], 'y': points_array[:, 1], 'value': u_array})

    return df




for i in range(1,6):
    filename = f"gaussian_30k_015_{i}/test0/crypt2_fk_model_sol_9900.pvtu" 
    print("reading file: {}".format(filename))
    df = get_obs_and_soln(filename)#, observation_coordinates)
    df.to_csv(f"test0_gaussian_30k015_1___{i}.csv", index=False)
    del(df)

#def main():
#    if len(sys.argv) != 3:
#        print("Usage: python script.py <input_filename> <output_csv_filename>")
#        sys.exit(1)
#
#    input_filename = sys.argv[1]
#    output_csv_filename = sys.argv[2]
#    #observation_coordinates = [(x1, y1), (x2, y2), ...]  # Replace with your actual observation coordinates
#    df = get_obs_and_soln("crypt2/fk_model_sol_0084.pvtu")#, observation_coordinates)
#    # Save the DataFrame to CSV
#    df.to_csv(output_csv_filename, index=False)
#
#if __name__ == "__main__":
#    main()
