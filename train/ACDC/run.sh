#!/bin/sh
timnow=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./history/$timnow"
mkdir $filename
cp *.py $filename
mv *.log ./history/
touch "$timnow.log"
for flod in $(seq 1 5)
do
    mkdir $filename/$flod
    python3 trainacdc.py --nunet 3 --base_filters 64 --batch_size 8 --timnow $timnow --training 1 --flod $flod
    mkdir $filename/$flod/d3_c64
    mv *.npy *.csv $filename/$flod/d3_c64
    mv ./result $filename/$flod/d3_c64
    python3 trainacdc.py --nunet 2 --base_filters 64 --batch_size 8 --timnow $timnow --training 1 --flod $flod
    mkdir $filename/$flod/d2_c64
    mv *.npy *.csv $filename/$flod/d2_c64
    mv ./result $filename/$flod/d2_c64
    python3 trainacdc.py --nunet 1 --base_filters 64 --batch_size 8 --timnow $timnow --training 1 --flod $flod
    mkdir $filename/$flod/d1_c64
    mv *.npy *.csv $filename/$flod/d1_c64
    mv ./result $filename/$flod/d1_c64
    
    python3 trainacdc.py --nunet 3 --base_filters 32 --timnow $timnow --training 1 --flod $flod
    mkdir $filename/$flod/d3_c32
    mv *.npy *.csv $filename/$flod/d3_c32
    mv ./result $filename/$flod/d3_c32
    python3 trainacdc.py --nunet 2 --base_filters 32 --timnow $timnow --training 1 --flod $flod
    mkdir $filename/$flod/d2_c32
    mv *.npy *.csv $filename/$flod/d2_c32
    mv ./result $filename/$flod/d2_c32
    python3 trainacdc.py --nunet 1 --base_filters 32 --timnow $timnow --training 1 --flod $flod
    mkdir $filename/$flod/d1_c32
    mv *.npy *.csv $filename/$flod/d1_c32
    mv ./result $filename/$flod/d1_c32
    cp -r ./pysave $filename/$flod/
done