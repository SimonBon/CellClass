#!/bin/bash

export PYTHONPATH="/src:$PYTHONPATH"

echo "Training Network"

    # p.add_argument("-d", "--down_steps", type=int, help="Number of downsampling steps with Input Size of 128x128 maximum of 7 to reach size of 1x1")
    # p.add_argument("-l", "--learning_rate", type=float, help="initial_learning_rate")
    # p.add_argument("-e", type=int, default=1000)
    # p.add_argument("-b", type=int, default=128)
    # p.add_argument("-s", type=str, help="directory to save the models to", default="/out")
    # p.add_argument("-", type=str, help="directory to all patches", default="/data")
    # p.add_argument("-p", type=str, help="Name of the positive Sample", default="S19")
    # p.add_argument("-negatives", type=str, help="Name of the negative Sample", default="S29")
    # p.add_argument("-log", type=bool, help="Define if logging should be done or not", default=True)

while [ $# -gt 0 ]; do
  case "$1" in
    --lr=*)
      lr="${1#*=}"
      ;;
    --epochs=*)
      epochs="${1#*=}"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      ;;
    --save_dir=*)
      save_dir="${1#*=}"
      ;;
    --patches_dir=*)
      patches_dir="${1#*=}"
      ;;
    --positives=*)
      positives="${1#*=}"
      ;;
    --negatives=*)
      negatives="${1#*=}"
      ;;
    --log=*)
      log="${1#*=}"
      ;;
    --n=*)
      n="${1#*=}"
      ;;
    *)
      printf "******************************\n"
      printf "* Error: $1 Invalid argument.*\n"
      printf "******************************\n"
      exit 1
  esac
  shift
done

#sh /src/CellClass/training.sh --d=3 --lr=0.01 --epochs=1000 --batch_size=128 --save_dir=/out --patches_dir=/data --positives=S19 --negatives=S29 --log=True --n=50
python3 /src/CellClass/train_docker.py -lr=$lr --epochs=$epochs --batch_size=$batch_size --save_dir=$save_dir --patches_dir=$patches_dir --positives=$positives --negatives=$negatives --log=$log -n=$n

