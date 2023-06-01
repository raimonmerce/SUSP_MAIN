import argparse
import subprocess
import yaml
import sys
import json
import os
import convert2Dto3D
import conver3DtoFront
from trescope import Trescope
from trescope.config import Scatter3DConfig
from trescope.config import FRONT3DConfig
from trescope.toolbox import simpleDisplayOutputs

def get_args(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )
    parser.add_argument(
        "--name_input_image",
        default="test.jpg",
        help="Name of input image"
    )
    parser.add_argument(
        "--path_to_input_image",
        default="input/",
        help="Path to input image"
    )
    parser.add_argument(
        "--name_output_file",
        default="test.json",
        help="Name of output file"
    )
    parser.add_argument(
        "--path_to_output_folder",
        default="output/",
        help="Path to output folder"
    )
    parser.add_argument(
        "--style",
        default="Modern",
        help="Style of furniture"
    )
    '''
    Styles:
    Modern
    Chinoiserie
    Kids
    European
    Japanese
    Southeast Asia
    Industrial
    American Country
    Vintage/Retro
    Light Luxury
    Mediterranean
    Korean
    New Chinese
    Nordic
    European Classic
    Others
    Ming Qing
    Neoclassical
    Minimalist
    '''
    parser.add_argument(
        "--mode_ss",
        default="visualize",
        help="Mode of scene synthesis: visualize"
    )
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        help="Path config file"
    )
    return parser.parse_args(argv)

def get_config(config_path):
    if isinstance(config_path, str) and os.path.isfile(config_path):
        if config_path.endswith('yaml'):
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                return config
        else:
            ValueError('Config file should be with the format of *.yaml')

def executeCFLandPB():
    commands = [
        #Commands for CFL
        {
            "cmd" : "cp " + args.path_to_input_image + args.name_input_image + " CFLPytorch/test/RGB/test.jpg",
            "wd" : "."
        },
        {
            "cmd" : "python3 test_TFCFL.py --conv_type Std --modelfile StdConvsTFCFL.pth",
            "wd" : "CFLPytorch"
        },
        {
            "cmd" : "mkdir tmp -p",
            "wd" : "."
        },
        {
            "cmd" : "cp CFLPytorch/test/CM_test/test_cmap.jpg tmp/test_cmap.jpg",
            "wd" : "."
        },
        {
            "cmd" : "cp CFLPytorch/test/EM_test/test_emap.jpg tmp/test_emap.jpg",
            "wd" : "."
        },
        {
            "cmd" : "cp " + args.path_to_input_image + args.name_input_image + " Panoramic-BlitzNet/dataset2/test.jpg",
            "wd" : "."
        },
        {
            "cmd" : "conda run -n " + config['conda_pbn'] + " python3 test.py PanoBlitznet",
            "wd" : "Panoramic-BlitzNet"
        },
        {
            "cmd" : "cp Panoramic-BlitzNet/Results/PanoBlitznet/test_segmentation_raw.png tmp/test_segmentation_raw.png",
            "wd" : "."
        },
        {
            "cmd" : "cp Panoramic-BlitzNet/Results/PanoBlitznet/test_boxes.json tmp/test_boxes.json",
            "wd" : "."
        }
    ]

    for cmd in commands:
        print(cmd)
        process = subprocess.Popen(cmd["cmd"], shell=True, cwd=cmd["wd"])    
        return_code = process.wait()
        while return_code != 0:
            return_code = process.wait()

def main(argv):
    global args 
    args = get_args(argv)
    global config
    config = get_config(args.config_path)
    #executeCFLandPB()
    dict3D = convert2Dto3D.convert()
    front3D = conver3DtoFront.convert(dict3D, args.style)
    jsonPath = args.path_to_output_folder + args.name_output_file
    with open(jsonPath, 'w') as f:
        json.dump(front3D, f)
    if (args.mode_ss == "visualize"):
        #Visualize Trescope
        Trescope().initialize(True, simpleDisplayOutputs(1, 1))
        Trescope().selectOutput(0).plotFRONT3D(jsonPath).withConfig(
            FRONT3DConfig()
            .view('top')
            .renderType('color')
            .shapeLocalSource(config['path_future'])
            .hiddenMeshes(['Ceiling', 'SlabTop', 'ExtrusionCustomizedCeilingModel']))
    elif (args.mode_ss == "completion"):
        #Scene Synthesis ATISS Scene Completion
        comand = "conda run -n " + config['conda_atiss'] + " python3 scene_completion.py " + config['path_to_config_yaml'] + " " + config['path_to_output_dir'] + " " + config['path_to_3d_future_pickled_data'] + " " + config['path_to_floor_plan_texture_images'] + " --weight_file " + config['path_to_weight_file'] + " --scene_id Bedroom-68718 --n_sequences 1 --with_rotating_camera"
        workinwDirectory = "ATISS/scripts"
        process = subprocess.Popen(comand, shell=True, cwd=workinwDirectory)    
        return_code = process.wait()
    else: 
        #Scene Synthesis ATISS Object Suggestions
        return False

if __name__ == "__main__":
    main(sys.argv[1:])