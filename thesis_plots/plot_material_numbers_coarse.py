import numpy as np
import scipy.fftpack

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys

# matplotlib.use('TkAgg')

RF_PREFIX = "coarse_RF-crypt1"
MAIN_FOLDER_NAME = "gaussian_coarse_1"

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

    if filter_material_number:
        return u_array, mat_array

    return u_array

def run_simulation(folder):
    overall = []
    mat_3_ = []
    mat_2_ = []
    mat_1_ = []

    FILENAME = "fk_model_sol_"

    # Observations

    for i in range(0, 12001, 200):
        if i == 0:
            print(f"First File: {MAIN_FOLDER_NAME}/{folder}/crypt[number?]_fk_model_sol_{i:04}.pvtu")
        print(f"{MAIN_FOLDER_NAME}/{folder}/file: {i}")
        points_array, mat_array = get_obs_and_soln(f"{MAIN_FOLDER_NAME}/{folder}/crypt0_fk_model_sol_{i:04}.pvtu", filter_material_number=[1, 2, 3])
        mat_1 = mat_array[0]
        mat_2 = mat_array[1]
        mat_3 = mat_array[2]
        overall.append(sum((points_array) / max(points_array)) / len(points_array))
        mat_2_.append(sum(mat_2 / max(mat_2)) / len(mat_2))
        mat_1_.append(sum(mat_1 / max(mat_1)) / len(mat_1))
        mat_3_.append(sum(mat_3 / max(mat_3)) / len(mat_3))

    crypt_label = ["Crypt Average", "Epithelial Layer", "ECM", "Lumen"]
    SAVE_NPY_PREFIX = f"{RF_PREFIX}_{folder}_"
    print(f"File prefix: {SAVE_NPY_PREFIX}")
    SAVE_PDF_NAME = SAVE_NPY_PREFIX + ".pdf"
    names = ["overall.npy", "mat1.npy", "mat2.npy", "mat3.npy"]
    save_names = [SAVE_NPY_PREFIX + name for name in names]

    np.save(save_names[0], overall)
    np.save(save_names[1], mat_1_)
    np.save(save_names[2], mat_2_)
    np.save(save_names[3], mat_3_)

    plt.style.use("seaborn-pastel")

    plt.plot(mat_1_, '-', label=crypt_label[3], color='blue', alpha=0.5)
    plt.plot(mat_2_, '-', label=crypt_label[1], color='green', alpha=0.5)
    plt.plot(mat_3_, '-', label=crypt_label[2], color='red', alpha=0.5)

    plt.plot(overall, '--', label=crypt_label[0], color='brown', alpha=0.7)

    plt.fill_between(range(len(overall)), overall, facecolor='brown', alpha=0.1)
    plt.fill_between(range(len(mat_1_)), mat_1_, facecolor='blue', alpha=0.1)
    plt.fill_between(range(len(mat_2_)), mat_2_, facecolor='green', alpha=0.1)
    plt.fill_between(range(len(mat_3_)), mat_3_, facecolor='red', alpha=0.1)

    legend_elements = [
        plt.Line2D([0], [0], color='brown', label=crypt_label[0]),
        plt.Line2D([0], [0], color='blue', label=crypt_label[3]),
        plt.Line2D([0], [0], color='green', label=crypt_label[1]),
        plt.Line2D([0], [0], color='red', label=crypt_label[2])
    ]

    plt.title(f"Evolution of Cancer Invasion in Crypt 2 - {folder}")
    plt.xticks([0, 10, 20, 30, 40, 50], labels=[0, 24, 48, 72, 96, 120])
    plt.ylabel("Probability of Invasion")
    plt.xlabel("Time (in months)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    print("Saving Plot...\n\n")
    plt.savefig(SAVE_PDF_NAME)
    plt.clf()

if __name__ == "__main__":
    folders = ['test0', 'test1', 'test2', 'test3']#, 'test4', 'test5', 'test6', 'test7']

    if len(sys.argv) == 2:
        specified_folder = sys.argv[1]
        if specified_folder in folders:
            run_simulation(specified_folder)
        else:
            print("Invalid folder name. Choose from: ", folders)
    else:
        for folder in folders:
            print(f"Reading from {folder}")
            run_simulation(folder)
