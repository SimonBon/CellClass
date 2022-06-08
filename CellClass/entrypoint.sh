
#!/bin/bash

export PYTHONPATH="/src:$PYTHONPATH"

echo "Extracting Patches from $1"

python3 src/CellClass/pipeline.py -i $1 -o /out -s $2