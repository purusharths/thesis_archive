from os import path, makedirs
import time
import subprocess
from lxml import etree

XML_PATH = "crypt1_fischerKPP_epi_BOTH.xml"
NPY_PATH = "/work/ws/hd_gs270-result-RandomFileds/RFs/"

INDEX = 2

alph_name = ["gaussian_smooth_wo", "gaussian_coarse_wo", "_MIX_ONLY+EXPO_AFTER_40"]
exec_name = ["wo_gr_highalpha", "wo_gr_lowalpha", "c111111"]

# ["30klow5", "30klow6", "30klow7", "30klow8", "30klow9"]
#random_filed_name  =  ["gaussian_30k_015_3", "gaussian_30k_015_6", "gaussian_30k_015_7", "gaussian_30k_015_8", "gaussian_30k_015_9", "30k_lowalpha1", "30k_lowalpha4", "30klow10", "30klow11", "30klow12", "matern_axip30k_8k2k_approx", "matern_axip20k_approx", "isotropic_30k_gaussian_axiparallel-0-1_0-3__4", "30khigh0", "30khigh1", "30khigh2"]
#["30khigh_5_12", "30khigh2", "30khigh1", "isotropic_30k_gaussian_axiparallel-0-1_0-3__2", "isotropic_30k_gaussian_axiparallel-0-1_0-3__4"]
random_filed_name  = ['30k_lowalpha1', '30k_lowalpha4', 'isotropic_expo_axi_0-1_0-02', 'isotropic_30k_gaussian_axiparallel-0-08_0-09__2', 'isotropic_30k_gaussian_axiparallel-0-08_0-09','expo2', 'expo4', 'expo6', 'expo7']
GENERATED_XML_suffix = f"{alph_name[INDEX]}_CRYPT1.xml"
PATH = "/work/ws/hd_gs270-result-RandomFileds/MIXED/crypt1/without_matern/"

SBATCH_CONTENTS = '''#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --mem=250gb

module load compiler/gnu
module load mpi/openmpi

mpirun -np 16 ./crypt1_fischerKPP_epi_BOTH {} {}'''

SBATCH_FILENAME = "submit_job.sh"

def change_xml_values(x_location_value, y_location_value, output_path_str, new_xml_filename):
    tree = etree.parse(XML_PATH)
    root = tree.getroot()
    initial_condition = root.find(".//InitialCondition")

    x_location = initial_condition.find("x_location")
    if x_location is not None:
        x_location.text = str(x_location_value)
    else:
        print("Error: x_location element not found in the XML.")

    y_location = initial_condition.find("y_location")
    if y_location is not None:
        y_location.text = str(y_location_value)
    else:
        print("Error: y_location element not found in the XML.")

    mesh = root.find(".//Mesh")
    output_path = mesh.find("OutputPath")
    if output_path is not None:
        output_path.text = str(output_path_str)
    else:
        print("Error: OutputPath element not found in the XML.")

    tree.write(new_xml_filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

def submit_sbatch(current_xml_filename, random_field_path, rf_name, subfolder_name):
    sbatch_filename = f'submit_job--{rf_name}__{subfolder_name}.sh'
    with open(sbatch_filename, 'w') as file:
        file.write(SBATCH_CONTENTS.format(path.join(PATH,current_xml_filename), path.join(NPY_PATH,rf_name)+".npy"))

    print("\tUpdated XML: {}\n \tSubmitting now...".format(current_xml_filename))
    # Uncomment the line below to enable sbatch submission
    subprocess.run(["sbatch", sbatch_filename])

def create_subfolders(base_path, subfolders):
    for subfolder in subfolders:
        folder_path = path.join(base_path, subfolder)
        if not path.exists(folder_path):
            makedirs(folder_path)

def generate_xml_for_subfolder(random_field_path, locations, subfolder, suffix, rf_name):
    for i, location in enumerate(locations):
        x_location_value, y_location_value = location
        new_xml_filename = f"{random_field_path}_{subfolder}_-{suffix}"
        output_path_and_filename = path.join(random_field_path, subfolder)
        print("=> outputpath: {}\n => newxml:{}".format(output_path_and_filename, new_xml_filename))
        change_xml_values(x_location_value, y_location_value, output_path_and_filename+"/", new_xml_filename)

def main(locations, subfolders, random_field_name, suffix, sleeptime=2):
    for rf_name in random_field_name:
        rf_path = path.join(PATH, rf_name)
        create_subfolders(rf_path, subfolders)
        idx = 0
        for subfolder in subfolders:
            random_field_path = path.join(PATH, rf_name)
            current_xml_filename = f"{random_field_path}_{subfolder}_-{suffix}"
            #print(random_field_path)

            print("Random Field: {}, Subfolder: {}".format(rf_name, subfolder))
            print(locations)
            print(idx)
            print("\tUpdate InitialCond: ({}, {}) \n\tWriting output to Path: {}\n\t----------".format(locations[idx][0], locations[idx][1], current_xml_filename))
            generate_xml_for_subfolder(random_field_path, [locations[idx]], subfolder, suffix, rf_name)
            submit_sbatch(current_xml_filename, random_field_path, rf_name, subfolder)
            print("\tSbatch submitted. \n\t(Sleeping for {}s) \n\n".format(sleeptime))
            idx+=1
            time.sleep(sleeptime)







if __name__ == "__main__":
    subfolders = ["test0", "test1", "test2"]#, "test3", "test4"]#, "test4"]#, "test5", "test6"]
    #crypt0_loc = [(0.047, 0.733), (0.157, 0.148), (0.493, 0.148), (0.606, 0.598), (0.936, 0.624)]
    crypt1_loc =  [(0.223, 0.282), (0.349, 0.634), (0.131, 0.789) ]   #[(0.223, 0.282), (0.131, 0.789), (0.349, 0.634),  (0.688, 0.889)]# (0.688,0.889)
    #crypt2_loc = [(0.090, 0.412), (0.356, 0.878), (0.552, 0.727), (0.702, 0.150), (0.670, 0.563), (0.802, 0.874), (0.844, 0.709)]
    main(crypt1_loc, subfolders, random_filed_name, GENERATED_XML_suffix, sleeptime=0)
    #main(crypt1_loc, folders, sleeptime=0.2)
    #print(alph_name[INDEX])
    #print(exec_name[INDEX])

