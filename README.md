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


