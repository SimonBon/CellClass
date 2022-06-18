import argparse
import os
import cv2
import pickle as pkl
from CellClass import MCImage, imread
from CellClass.Segment import Segmentation
from CellClass.process_masks import get_cell_patches


# get arguments for preparation of images to save them in BGR and tif format. 
def parse():

    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in_path")
    p.add_argument("-o", "--out_path")
    p.add_argument("-s", "--sample_name")
    p.add_argument("--algorithm", default="cellpose")

    return p.parse_args()

#images finden die dem gegebenem Pattern entsprechen
def find_matching_images(dir, pattern):

    matches = [x for x in os.listdir(dir) if f"{pattern}_" in x and not x.startswith(".")]
    return matches

if __name__=="__main__":

    args = parse()
    
    if not os.path.isdir(args.in_path):
        print(f"{args.in_path} does not exist!")
        exit()

    names = find_matching_images(args.in_path, args.sample_name)

    S = Segmentation(args.algorithm)

    # for each sample extract the patches
    for n in names:

        sample_name = n.split(".")[0]
        print("Working on: ", sample_name)
        if os.path.isfile(os.path.join(args.out_path, f'{sample_name}.ptch')):
            print(sample_name + " already exists!")
            continue

        
        img = imread(os.path.join(args.in_path, n))
        MCIm = MCImage(img, scheme="BGR")
        MCIm.normalize()

        
        if args.algorithm == "deepcell":
            im, res, o = S(MCIm.B, return_outline=True, image_mpp=0.4)
        else:
            im, res, o = S(MCIm.B, return_outline=True)

        cv2.imwrite(os.path.join(args.out_path, f'masks/{sample_name}_masks.tif'), res.astype("uint16"))
        print("Saved Masks for " + sample_name)

        patches = get_cell_patches(MCIm, res, size=128)

        with open(os.path.join(args.out_path, f'patches/{sample_name}.ptch'), 'wb+') as f:
            pkl.dump(patches, f)
        print("Saved Patches for " + sample_name)

            
        cv2.imwrite(os.path.join(args.out_path, f'overlays/{sample_name}_o.jpg'), (o.astype("float32")*255).astype("uint8"))
        print("Saved Overlay for " + sample_name)


