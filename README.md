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

--dataset
ucf101
--n_finetune_classes
101
--ft_begin_index
5
--weight_decay
1e-3
--learning_rate
1e-1
--model
resnet
--resnet_shortcut
A
--model_depth
18
--n_classes
400
--batch_size
24
--n_threads
4
--checkpoint
5
--manual_seed
1
--n_epochs
60
--test
--test_subset
val
# acqusition args
     # acqusition args
    parser.add_argument('--init_train_size', type=int, default=101)
    parser.add_argument('--max_train_size', type=int, default=1010)
    # selection of the initial training dataset: random, uniform_random, same (same as active learning selection)
    parser.add_argument('--init_selection', type=str, default='uniform_random')
    # random or var_ratio (random means no uncertainty)
    parser.add_argument('--score_func', type=str, default='var_ratio')
    # unsupervised-kmeans, unsupervised-kmedoids, unsupervised-spectral, unsupervised-network, unsupervised-ds-svm ,
    # labels, none
    parser.add_argument('--clustering', type=str, default='none')
    # e_optimal, a_optimal, d_optimal, inv_cond, IPM, ds3, kmedoids, none (just score)
    parser.add_argument('--optimality', type=str, default='IPM')

    parser.add_argument('--n_pool', type=int, default=101)

    parser.add_argument('--n_clust', type=int, default=101)
    parser.add_argument('--n_pool_clust', type=int, default=1)

    parser.add_argument('--alpha', type=float, default=1)  # alpha = 1 (just eig score)
    parser.add_argument('--alpha_decay', type=float, default=0.95)


    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--MC_runs', type=int, default=10)



