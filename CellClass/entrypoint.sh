#!/bin/bash

export PYTHONPATH="/src:$PYTHONPATH"

echo "Extracting Patches from $1"


#python3 src/CellClass/pipeline.py -i input_path (hier /tmp) -s Sample (z.B. S19) --algorithm (deepcell / cellpose)
python3 src/CellClass/pipeline.py -i $1 -o /out -s $2 --algorithm $3
