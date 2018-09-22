export PYTHONPATH="."

dataset_name=cifar10

dataloader_batch_size=32

network_name=resnet
network_args_1=10
network_args_2=50
network_args_3=2

for i in {1..5}
do
    experiment_folder=results/${dataset_name}/${network_name}/${network_args_1}_${network_args_2}_${network_args_3}/try${i}

    python opts.py \
           --dataloader-batch-size ${dataloader_batch_size} \
           --dataset-name ${dataset_name} \
           --network-name ${network_name} \
           --network-args ${network_args_1} ${network_args_2} ${network_args_3} \
           --experiment-folder ${experiment_folder}

    python main.py ${experiment_folder}/opts.txt
done
