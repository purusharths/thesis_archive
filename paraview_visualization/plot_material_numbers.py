import numpy as np
import scipy.fftpack

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import vtk
from vtk.util.numpy_support import vtk_to_numpy
matplotlib.use('TkAgg')


def get_val(x, y, points_array, u_array):
    # Convert x, y to the index in points_array
    idx = np.argmin((points_array[:, 0] - x)**2 + (points_array[:, 1] - y)**2)
    return u_array[idx]

def get_loc(x, y, points_array, u_array):
    # Convert x, y to the index in points_array
    idx = np.argmin((points_array[:, 0] - x)**2 + (points_array[:, 1] - y)**2)
    return tuple(points_array[idx][:2])


def get_material_number_data(mat_array, list_of_matnums, u_array, output):

    main_list_value_array = []
    main_list_mat_coordinate = []
    for matnum in list_of_matnums:
        material_number_to_filter = matnum
        filtered_u_array = u_array[mat_array == material_number_to_filter]
        filtered_coordinates = vtk_to_numpy(output.GetPoints().GetData())[mat_array == material_number_to_filter]
        main_list_mat_coordinate.append(filtered_coordinates)
        main_list_value_array.append(filtered_u_array)

    return main_list_value_array, main_list_mat_coordinate


def get_obs_and_soln(filename, filter_material_number=None): #, observation_points):
    # Reading PVTU
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
    
    filtered_u_array = []
    mat_array = []
    if filter_material_number is not None:
        material_array = None
        for i in range(num_arrays):
            array_name = point_data_output.GetArrayName(i)
            # print(array_name)
            if array_name == 'Material Id':
                material_array = vtk_to_numpy(point_data_output.GetArray(i))
                break

        if material_array is None:
            print("Material numbers not found in the dataset.")
            exit()
    mat_array, mat_coords = get_material_number_data(material_array, filter_material_number, u_array, output)


    # # Convert the data to NumPy arrays
    # points_array = vtk_to_numpy(output.GetPoints().GetData())

    # # Plot the data
    # plt.scatter(filtered_coordinates[:, 0], filtered_coordinates[:, 1], c=filtered_u_array, cmap='jet', marker=".")
    # plt.colorbar()
    # plt.savefig("figure.png")
    # plt.show()


    # value = []
    # location = []
    # Example usage
    # for observation_point in observation_points:
    #     x = observation_point[0] 
    #     y = observation_point[1]

    #     value.append(get_val(x, y, points_array, u_array))
    #     location.append(get_loc(x, y, points_array, u_array))

    #print(f"Value at: {value}")
    #print(f"Location at: {location}")
    #return value, location, u_array
    if filter_material_number:
        return u_array, mat_array
    
    return u_array
    #observation_coordinates = [(0.4, 0.4), (0.3,0.3), (0.1,0.2), (0.5,0.5), (0.8,0.2), (0.2,0.1)] # randomly chosen points


## FEM evaluation
#observation_coordinates = [(0.4, 0.4), (0.3,0.3), (0.1,0.2), (0.5,0.5), (0.8,0.2), (0.2,0.1)] # randomly chosen points

overall = []
mat_3_ = []
mat_2_ = []
mat_1_ = []

FILENAME = "fk_model_sol_"
# ## Observations
for i in range(0,1001,20):
    print(i)
    points_array, mat_array = get_obs_and_soln("fk_model_sol_{0:04}.pvtu".format(i) , filter_material_number=[1,2,3])
    # mat_1 = get_obs_and_soln("fk_epi_BOTH_{0:04}.pvtu".format(i) , 1)
    # mat_2 = get_obs_and_soln("fk_epi_BOTH_{0:04}.pvtu".format(i) , 2)
    # mat_3 = get_obs_and_soln("fk_epi_BOTH_{0:04}.pvtu".format(i) , 3)

    #overall.append((points_array)/(sum(points_array)/len(points_array)))
    #mat_2_.append((mat_2)/(sum(mat_2)/len(mat_2)))
    #mat_1_.append((mat_1)/(sum(mat_1)/len(mat_1)))
    #mat_3_.append((mat_3)/(sum(mat_3)/len(mat_3)))
    mat_1 = mat_array[0]
    mat_2 = mat_array[1]
    mat_3 = mat_array[2]
    overall.append(sum((points_array)/max(points_array))/len(points_array))
    mat_2_.append(sum(mat_2/max(mat_2))/len(mat_2))
    mat_1_.append(sum(mat_1/max(mat_1))/len(mat_1))
    mat_3_.append(sum(mat_3/max(mat_3))/len(mat_3))



plt.plot(overall, '-',label = "Overall")
plt.plot(mat_1_, '.', label = "Lumen")
plt.plot(mat_2_, '.', label = "Epithelial")
plt.plot(mat_3_, '.', label = "ECM")
plt.grid(True)
plt.legend()
plt.savefig("a.pdf")
plt.show(block=True)