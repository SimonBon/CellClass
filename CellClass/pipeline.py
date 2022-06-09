import argparse
import os
import pickle as pkl
from CellClass import MCImage, imread
from CellClass.Segment import Segmentation
from CellClass.process_masks import get_cell_patches


def parse():

    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in_path")
    p.add_argument("-o", "--out_path")
    p.add_argument("-s", "--sample_name")

    return p.parse_args()

def find_matching_images(dir, pattern):

    matches = [x for x in os.listdir(dir) if f"{pattern}_" in x and not x.startswith(".")]
    return matches

if __name__=="__main__":

    args = parse()
    
    if not os.path.isdir(args.in_path):
        print(f"{args.in_path} does not exist!")
        exit()

    names = find_matching_images(args.in_path, args.sample_name)

    for n in names:

        sample_name = n.split(".")[0]
        print("Working on: ", sample_name)
        if os.path.isfile(os.path.join(args.out_path, f'{sample_name}.ptch')):
            print(sample_name + " already exists!")
            continue

        
        img = imread(os.path.join(args.in_path, n))
        MCIm = MCImage(img, scheme="BGR")
        MCIm.normalize()

        S = Segmentation()
        _, res = S(MCIm.B, return_outline=False)

        patches = get_cell_patches(MCIm, res, size=128)

        with open(os.path.join(args.out_path, f'{sample_name}.ptch'), 'wb+') as f:
            print("Saved " + sample_name)
            pkl.dump(patches, f)


