import os
import pickle as pkl
from typing_extensions import dataclass_transform
from tqdm import tqdm

def define_out_dir(out_dir):
    
    if not os.path.isdir(out_dir):
        decision = input(f"{out_dir} does not exist, do you want to create it? [y/n]")
        if decision.lower() == "y":
            os.makedirs(out_dir)
            print(f"{out_dir} created!")
        else:
            print(f"{out_dir} not created!")

def create_single_patches(in_dir: str, out_dir: str, sample: str, n=None):
    
    define_out_dir(out_dir)
    sample_str = sample + "_"
    files = [x for x in os.listdir(in_dir) if sample_str in x]
    
    if isinstance(n, int):
        print(f"Saving {n} files!")
        gen = tqdm(files[:n])
    else:
        gen = tqdm(files)
        
    for file in gen:
        
        subsample_name = file.split(".")[0]
        if os.path.isfile(os.path.join(out_dir, f"{subsample_name}_0.ptch")):
            continue
        
        with open(os.path.join(in_dir, file), "rb") as fin:
            patch_list = pkl.load(fin)
            
            for i, patch in enumerate(patch_list):
                with open(os.path.join(out_dir, f"{subsample_name}_{i}.ptch"), "wb+") as fout:
                    pkl.dump(patch, fout)


            
    