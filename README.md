# Active Learning using Iterative Projection and Matching
Implementation of the active learning experiment on UCF-101 video dataset proposed in: 

Alireza Zaeemzadeh, Mohsen Joneidi ( shared first authorship) , Nazanin Rahnavard, Mubarak Shah: Iterative Projection and Matching: Finding Structure-preserving Representatives and Its Application to Computer Vision. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
[link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zaeemzadeh_Iterative_Projection_and_Matching_Finding_Structure-Preserving_Representatives_and_Its_Application_CVPR_2019_paper.pdf)


## Requirements

Most of the training code is used form [here.](https://github.com/kenshohara/3D-ResNets-PyTorch)

Tested on:
- Python 2.7
- cuda 9.1 (2 GPUs)
- torch 0.4.1
- torchvision 0.2.1
- irlbpy 0.1.0 [code](https://github.com/bwlewis/irlbpy)

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```


### Dataset
* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```

### Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing). 

Info on pretraining available [here](https://github.com/kenshohara/3D-ResNets-PyTorch).


## Sample Script
```bash
python main.py --root_path data/ --video_path frames/ --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results/ --pretrain_path pretrained/resnet-18-kinetics.pth --model resnet --resnet_shortcut A --model_depth 18 --test --test_subset val
```
A 3DResNet18 model, pretrained on Kinetics, is fine tuned at each active learning cycle and is used to select the most informative samples.



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

[Standalone version of the data selection algorithm](https://github.com/zaeemzadeh/IPM)
