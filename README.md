```bash
conda install pytorch torchvision cuda80 -c soumith
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```


### UCF-101

Download http://crcv.ucf.edu/data/UCF101.php

python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory

python utils/n_frames_ucf101_hmdb51.py jpg_video_directory


~/
  data/
      frames/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    test.json


## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing). 

Info on pretraining available [here](https://github.com/kenshohara/3D-ResNets-PyTorch).


## Parameters

--root_path /home/alireza/Desktop/Deep-Learning/ucf101-data --video_path frames/ --annotation_path /home/alireza/Desktop/Deep-Learning/ucf101-data/ucfTrainTestlist/ucf101_01.json --result_path /home/alireza/Desktop/Deep-Learning/ucf101-data/results/ --pretrain_path /home/alireza/Desktop/Deep-Learning/UCF101-Active-Learning/pretrained/resnet-18-kinetics.pth --dataset ucf101 --n_finetune_classes 101 --ft_begin_index 5 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 1 --n_epochs 50

# acqusition args
    parser.add_argument('--init_train_size', type=int, default=101)
    parser.add_argument('--max_train_size', type=int, default=505)
    # selection of the initial training dataset: random, uniform_random, same (same as active learning selection)
    parser.add_argument('--init_selection', type=str, default='uniform_random')
    # random or var_ratio (random means no uncertainty) TODO: implement var_ratio
    parser.add_argument('--score_func', type=str, default='random')
    # unsupervised, labels, none
    parser.add_argument('--clustering', type=str, default='labels')
    # e_optimal, a_optimal, d_optimal, inv_cond, IPM, none (just score)
    parser.add_argument('--optimality', type=str, default='none')

    parser.add_argument('--n_pool', type=int, default=101)

    parser.add_argument('--n_clust', type=int, default=101)
    parser.add_argument('--n_pool_clust', type=int, default=1)

    parser.add_argument('--alpha', type=float, default=1)               # alpha = 1 (just eig score)
    parser.add_argument('--alpha_decay', type=float, default=1)

    args = parser.parse_args()



