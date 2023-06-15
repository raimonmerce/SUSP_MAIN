import os
import cv2 
import Perspective.lib.Equirec2Perspec as E2P
import Perspective.lib.Perspec2Equirec as P2E
import Perspective.lib.multi_Perspec2Equirec as m_P2E
import glob
import sys
import argparse

def equir2pers_save(input_img, output_img, FOV, theta, phi, height, width):
    equ = E2P.Equirectangular(input_img)    # Load equirectangular image

    img = equ.GetPerspective(FOV, theta, phi, height, width)  # Specify parameters(FOV, theta, phi, height, width)
    cv2.imwrite(output_img, img)

def equir2pers(input_img, output_img, FOV, theta, phi, height, width):
    equ = E2P.Equirectangular(input_img)    # Load equirectangular image

    img = equ.GetPerspective(FOV, theta, phi, height, width)  # Specify parameters(FOV, theta, phi, height, width)
    cv2.imwrite(output_img, img)
    return img

def getArgs(argv):

    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #

    parser = argparse.ArgumentParser(
        description="Generate perspective image prom equirectangular image"
    )

    parser.add_argument(
        "input_image",
        default='./panorama/test.jpg',
        help="Path to the input image"
    )

    parser.add_argument(
        "output_directory",
        default='./example/perspective',
        help="Path to the output directory"
    )

    parser.add_argument(
        "FOV",
        default=120,
        type=int,
        help="FOV camera"
    )

    parser.add_argument(
        "theta",
        default=0,
        type=int,
        help="Theta camera"
    )

    parser.add_argument(
        "phi",
        default=0,
        type=int,
        help="Phi camera"
    )

    parser.add_argument(
        "height",
        default=1280,
        type=int,
        help="height output picture"
    )

    parser.add_argument(
        "width",
        default=1280,
        type=int,
        help="width output picture"
    )

    args = parser.parse_args(argv)
    
    input_img = args.input_image
    output_dir = args.output_directory
    FOV = args.FOV
    theta = args.theta
    phi = args.phi
    height = args.height
    width = args.width

    equir2pers(input_img, output_dir, FOV, theta, phi, height, width)

if __name__ == '__main__':
    #python3 equir2pers.py ./panorama/test.jpg ./example/perspective 120 0 0 1280 1280
    getArgs(sys.argv[1:])