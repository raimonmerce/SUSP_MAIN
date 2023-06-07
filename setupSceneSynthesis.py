import subprocess
import csv

def setup():
    '''
    #Preprocess new json
    path_to_output_tmp_dir = "../../ATISS_extra/output_tmp"
    path_to_tmp_3d_front_dataset_dir = "../../output"
    path_to_3d_future_dataset_dir = "../../../../../../media/raimon/SSD/3Dfront/3D-FUTURE-model"
    path_to_3d_future_model_info = "../../../../../../media/raimon/SSD/3Dfront/3D-FUTURE-model/model_info.json"
    path_to_floor_plan_texture_images = "../demo/floor_plan_texture_images"
    command_preprocess = "python3 preprocess_data.py " + path_to_output_tmp_dir + " " + path_to_tmp_3d_front_dataset_dir + " " + path_to_3d_future_dataset_dir + " " + path_to_3d_future_model_info + " " + path_to_floor_plan_texture_images + " --dataset_filtering threed_front_bedroom"

    #Copy json on output folder
    path_to_preprocessed = path_to_output_tmp_dir + "/test_generate"
    path_to_output_dir = "../../ATISS_extra/output"
    command_copy = "cp -r " + path_to_preprocessed + " " + path_to_output_dir

    #Modify .csv file
    #new_row = ['generate', 'train']
    #csv_file_path = 'path/to/file.csv'
    '''

    working_directory = "ATISS/scripts"
    path_to_json = "../../output/test.json"
    path_to_3d_front_dataset_dir = "../../../../../../media/raimon/SSD/3Dfront/3D-FRONT" 
    command_copy = "cp " + path_to_json + " " + path_to_3d_front_dataset_dir

    process = subprocess.Popen(command_copy, shell=True, cwd=working_directory)    
    return_code = process.wait()
    while return_code != 0:
        return_code = process.wait()
    
    #Preprocess pickle
    path_to_output_pickle_dir = "../../ATISS_extra/pickle_tmp/"
    path_csv = "../config/bedroom_threed_front_splits_2.csv"
    path_to_3d_future_dataset_dir = "../../../../../../media/raimon/SSD/3Dfront/3D-FUTURE-model"
    path_to_3d_future_model_info = "../../../../../../media/raimon/SSD/3Dfront/3D-FUTURE-model/model_info.json"
    command_pickle = "python3 pickle_threed_future_dataset.py " + path_to_output_pickle_dir + " " + path_to_3d_front_dataset_dir  + " " + path_to_3d_future_dataset_dir + " " + path_to_3d_future_model_info + " --dataset_filtering threed_front_bedroom --annotation_file " + path_csv

    process = subprocess.Popen(command_pickle, shell=True, cwd=working_directory)    
    return_code = process.wait()
    while return_code != 0:
        return_code = process.wait()