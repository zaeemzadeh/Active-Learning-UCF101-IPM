# Active Learning using Iterative Projection and Matching
implementation of the active learning algorithm proposed in: 

Alireza Zaeemzadeh, Mohsen Joneidi ( shared first authorship) , Nazanin Rahnavard, Mubarak Shah: Iterative Projection and Matching: Finding Structure-preserving Representatives and Its Application to Computer Vision. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
[link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zaeemzadeh_Iterative_Projection_and_Matching_Finding_Structure-Preserving_Representatives_and_Its_Application_CVPR_2019_paper.pdf)


## Citing IPM
If you use IPM in your research, please use the following BibTeX entry.
```
@inproceedings{zaeemzadeh2019ipm,
    title = {{Iterative Projection and Matching: Finding Structure-preserving Representatives and Its Application to Computer Vision}},
    year = {2019},
    booktitle = {Computer Vision and Pattern Recognition, 2019. CVPR 2019. IEEE Conference on},
    author = {Zaeemzadeh, Alireza and Joneidi, Mohsen and Rahnavard, Nazanin and Shah, Mubarak}
}
```


## Project Webpages
[Presentation ](https://youtu.be/OFe5z5fMUGc)

[UCF Center for Research in Computer Vision (CRCV)](https://www.crcv.ucf.edu/home/projects/iterative-projection-and-matching/)

[UCF Communications and Wireless Networks Lab (CWNlab)](http://cwnlab.eecs.ucf.edu/ipm/)

For inquiries, please contact zaeemzadeh -at- knights.ucf.edu.


## Requirements

Tested on:
- Python 2.7
- torch 0.4.1
- torchvision 0.2.1
- irlbpy 0.1.0 [code](https://github.com/bwlewis/irlbpy)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```


### Dataset

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


### Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing). 

Info on pretraining available [here](https://github.com/kenshohara/3D-ResNets-PyTorch).


## Sample Script
```bash
python main.py --root_path data/ --video_path frames/ --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results/ --pretrain_path pretrained/resnet-18-kinetics.pth --n_finetune_classes 101 --ft_begin_index 5 --weight_decay 1e-3 --learning_rate 1e-1 --model resnet --resnet_shortcut A --model_depth 18 --n_classes 400 --batch_size 24 --n_threads 4 --checkpoint 5 --manual_seed 2 --n_epochs 60 --test --test_subset val
```