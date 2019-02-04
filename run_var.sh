#/home/alireza/miniconda2/bin/python /home/alireza/Desktop/I3D-RGB-branch/main.py --root_path /home/alireza/Desktop/I3D-RGB-branch/data --video_path frames/ --annotation_path /home/alireza/Desktop/I3D-RGB-branch/data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/I3D-RGB-branch/data/results/ --pretrain_path /home/alireza/Desktop/I3D-RGB-branch/pretrained/resnet-18-kinetics.pth --dataset ucf101 --n_finetune_classes 101 --ft_begin_index 5 --weight_decay 1e-3 --learning_rate 1e-1 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 1 --n_epochs 60 --test --test_subset val --clustering none --init_selection uniform_random --score_func var_ratio --optimality IPM --alpha 1 --alpha_decay 0.95 --dropout_rate 0.2 --MC_runs 10 > IPM_VAR1.txt

/home/alireza/miniconda2/bin/python /home/alireza/Desktop/I3D-RGB-branch/main.py --root_path /home/alireza/Desktop/I3D-RGB-branch/data --video_path frames/ --annotation_path /home/alireza/Desktop/I3D-RGB-branch/data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/I3D-RGB-branch/data/results/ --pretrain_path /home/alireza/Desktop/I3D-RGB-branch/pretrained/resnet-18-kinetics.pth --dataset ucf101 --n_finetune_classes 101 --ft_begin_index 5 --weight_decay 1e-3 --learning_rate 1e-1 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 1 --n_epochs 60 --test --test_subset val --clustering none --init_selection uniform_random --score_func random --optimality IPM --alpha 1 --alpha_decay 1 --dropout_rate 0 --MC_runs 1 > IPM1.txt

#/home/alireza/miniconda2/bin/python /home/alireza/Desktop/I3D-RGB-branch/main.py --root_path /home/alireza/Desktop/I3D-RGB-branch/data --video_path frames/ --annotation_path /home/alireza/Desktop/I3D-RGB-branch/data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/I3D-RGB-branch/data/results/ --pretrain_path /home/alireza/Desktop/I3D-RGB-branch/pretrained/resnet-18-kinetics.pth --dataset ucf101 --n_finetune_classes 101 --ft_begin_index 5 --weight_decay 1e-3 --learning_rate 1e-1 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 1 --n_epochs 60 --test --test_subset val --clustering none --init_selection uniform_random --score_func var_ratio --optimality none --alpha 1 --alpha_decay 1 --dropout_rate 0.2 --MC_runs 10 > VAR1.txt

/home/alireza/miniconda2/bin/python /home/alireza/Desktop/I3D-RGB-branch/main.py --root_path /home/alireza/Desktop/I3D-RGB-branch/data --video_path frames/ --annotation_path /home/alireza/Desktop/I3D-RGB-branch/data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/I3D-RGB-branch/data/results/ --pretrain_path /home/alireza/Desktop/I3D-RGB-branch/pretrained/resnet-18-kinetics.pth --dataset ucf101 --n_finetune_classes 101 --ft_begin_index 5 --weight_decay 1e-3 --learning_rate 1e-1 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 1 --n_epochs 60 --test --test_subset val --clustering none --init_selection uniform_random --score_func random --optimality kmedoids --alpha 1 --alpha_decay 1 --dropout_rate 0 --MC_runs 1 > kmedoids1.txt

/home/alireza/miniconda2/bin/python /home/alireza/Desktop/I3D-RGB-branch/main.py --root_path /home/alireza/Desktop/I3D-RGB-branch/data --video_path frames/ --annotation_path /home/alireza/Desktop/I3D-RGB-branch/data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/I3D-RGB-branch/data/results/ --pretrain_path /home/alireza/Desktop/I3D-RGB-branch/pretrained/resnet-18-kinetics.pth --dataset ucf101 --n_finetune_classes 101 --ft_begin_index 5 --weight_decay 1e-3 --learning_rate 1e-1 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 1 --n_epochs 60 --test --test_subset val --clustering none --init_selection uniform_random --score_func random --optimality ds3 --alpha 1 --alpha_decay 1 --dropout_rate 0 --MC_runs 1 > ds31.txt



