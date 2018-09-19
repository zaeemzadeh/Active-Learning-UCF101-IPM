#For UCF-101, validation set is test set.
#Tuning on Kinetics can reset learning rate, Resume training cannot since it will use the saved optimizer.

#training from scratch
python main.py --root_path /home/alireza/Desktop/I3D-RGB-branch/data --video_path frames/ --annotation_path /home/alireza/Desktop/I3D-RGB-branch/data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/I3D-RGB-branch/data/results/ --dataset ucf101 --model resnet --resnet_shortcut B --model_depth 18 --n_classes 101 --batch_size 8 --n_threads 4 --checkpoint 5 --manual_seed 1  #--no_val #--test --test_subset val

#tuning from pretrained model (with validation)
#python main.py --root_path /home/student/UCF101/ --video_path frames/ --annotation_path /home/student/UCF101/ucfTrainTestlist/ucf101_01.json --dataset ucf101 --n_finetune_classes 101 --pretrain_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics.pth --ft_begin_index 4 --batch_size 8 --n_threads 8 --checkpoint 10 --manual_seed 1 --model resnet --resnet_shortcut A --model_depth 18 --learning_rate 0.1 --result_path /home/student/3D-ResNets-PyTorch/results/ 

#tuning from pretrained model (without validation to save space)
#python main.py --root_path /home/student/UCF101/ --video_path frames/ --annotation_path /home/student/UCF101/ucfTrainTestlist/ucf101_01.json --dataset ucf101 --n_finetune_classes 101 --pretrain_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics.pth --ft_begin_index 4 --batch_size 8 --n_threads 8 --checkpoint 10 --manual_seed 1 --model resnet --resnet_shortcut A --model_depth 18 --learning_rate 0.1 --result_path /home/student/3D-ResNets-PyTorch/results/ --no_val 

#continue training on fine-tuned model (without validation to save space)
#python main.py --root_path /home/student/UCF101/ --video_path frames/ --annotation_path /home/student/UCF101/ucfTrainTestlist/ucf101_01.json --dataset ucf101 --pretrain_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics.pth --n_classes 400 --n_finetune_classes 101 --ft_begin_index 4 --resume_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics-ucf101_split1.pth --batch_size 64 --n_threads 8 --checkpoint 1 --manual_seed 1  --model resnet --model_depth 18 --resnet_shortcut A --result_path /home/student/3D-ResNets-PyTorch/results/ --n_epochs 400 --no_val 

#when we have the pretrained models, clip-level accuracy on fine-tuned model
#python main.py --root_path /home/student/UCF101/ --video_path frames/ --annotation_path /home/student/UCF101/ucfTrainTestlist/ucf101_01.json --dataset ucf101 --pretrain_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics.pth --n_classes 400 --n_finetune_classes 101 --ft_begin_index 4 --resume_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics-ucf101_split1.pth --batch_size 60 --n_threads 8 --checkpoint 10 --manual_seed 1  --model resnet --model_depth 18 --resnet_shortcut A --result_path /home/student/3D-ResNets-PyTorch/results/ --n_epochs 301 --no_train #--no_val --test --test_subset val 


#when we have the pretrained models, video-level accuracy on fine-tuned model (the best model we obtained from clip-level accuracy)
#python main.py --root_path /home/student/UCF101/ --video_path frames/ --annotation_path /home/student/UCF101/ucfTrainTestlist/ucf101_01.json --dataset ucf101 --pretrain_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics.pth --n_classes 400 --n_finetune_classes 101 --ft_begin_index 4 --resume_path /home/student/3D-ResNets-PyTorch/pretrained/resnet-18-kinetics-ucf101_split1.pth --batch_size 60 --n_threads 8 --checkpoint 10 --manual_seed 1  --model resnet --model_depth 18 --resnet_shortcut A --result_path /home/student/3D-ResNets-PyTorch/results/ --n_epochs 301 --no_train --no_val --test --test_subset val 



