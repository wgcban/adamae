# Kinetics400 dataset (https://www.deepmind.com/open-source/kinetics)

## Donwloading
To download the kinetics-400 dataset follow the steps in the repo: https://github.com/wgcban/kinetics-dataset

##
Extract train/vaal/test/currupted files

# Resize videos
Previous studies found that resizing the short side length to **320 pixels** while keeping the aspect ratio have given better results (see: https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn#kinetics-400-data-benchmark-8-gpus-resnet50-imagenet-pretrain-3-segments, and https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly#kinetics-400-data-benchmark)

Following this, use the following script to resize the kinetics-400 videos: https://github.com/open-mmlab/mmaction2/blob/master/tools/data/resize_videos.py

Train
```
python mmaction2/tools/data/resize_videos.py kinetics-dataset/k400/train/ k400_320p/ --to-mp4 --scale 320 --num-worker 8 --level 1
```

Val
```
python mmaction2/tools/data/resize_videos.py kinetics-dataset/k400/val/ k400_320p/ --to-mp4 --scale 320 --num-worker 8 --level 1
```

Test
```
python mmaction2/tools/data/resize_videos.py kinetics-dataset/k400/test/ k400_320p/ --to-mp4 --scale 320 --num-worker 8 --level 1
```

Corrupted files
```
python mmaction2/tools/data/resize_videos.py kinetics-dataset/k400/replacement/replacement_for_corrupted_k400/ k400_320p/ --to-mp4 --scale 320 --num-worker 8 --level 1
```

## Generate annotations
Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes train.csv, val.csv and test.csv ( here test.csv is the same as val.csv). The format of *.csv file is like:

```
dataset_root/video_1.mp4  label_1
dataset_root/video_2.mp4  label_2
dataset_root/video_3.mp4  label_3
...
dataset_root/video_N.mp4  label_N
```

For that run the following code (assume you have tran/val/test csv files (https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz) and labels.csv (https://gist.githubusercontent.com/willprice/f19da185c9c5f32847134b87c1960769/raw/9dc94028ecced572f302225c49fcdee2f3d748d8/kinetics_400_labels.csv)) downloaded and saved in ./kinetics400:

```
import csv
import os

# get the class number
labels_400 = {}
with open("kinetics400/kinetics_400_labels.csv") as f_labels:
    reader_labels = csv.reader(f_labels, delimiter="\t")
    next(reader_labels)
    for i, line in enumerate(reader_labels):
        n = line[0].split(",")[1]
        v = line[0].split(",")[0]
        labels_400[n] = v

# Set datapath and CSV save path
DATA_PATH = '/data/wbandar1/datasets/kinetics/k400_320p/'
CSV_WRITE_PATH = '/data/wbandar1/datasets/kinetics/k400_320p_lists'

#Generating "train" split
with open(os.path.join(CSV_WRITE_PATH,'kinetics-train.csv'),'w') as f_csv:
    writer=csv.writer(f_csv, delimiter=' ',lineterminator='\n',)
    
    with open("kinetics400/train.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(',')
            video_path = DATA_PATH+line[1]+"_"+line[2].zfill(6)+"_"+line[3].zfill(6)+".mp4"
            video_lbl_num = labels_400[line[0]]
            print(video_path, video_lbl_num)
            writer.writerow([video_path, video_lbl_num])
f_csv.close()
f.close()

#Generating "val" split
with open(os.path.join(CSV_WRITE_PATH,'kinetics-val.csv'),'w') as f_csv:
    writer=csv.writer(f_csv, delimiter=' ',lineterminator='\n',)
    
    with open("kinetics400/validate.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(',')
            video_path = DATA_PATH+line[1]+"_"+line[2].zfill(6)+"_"+line[3].zfill(6)+".mp4"
            video_lbl_num = labels_400[line[0]]
            print(video_path, video_lbl_num)
            writer.writerow([video_path, video_lbl_num])
f_csv.close()
f.close()

#Generating "train" split
with open(os.path.join(CSV_WRITE_PATH,'kinetics-test.csv'),'w') as f_csv:
    writer=csv.writer(f_csv, delimiter=' ',lineterminator='\n',)
    
    with open("kinetics400/test.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(',')
            video_path = DATA_PATH+line[1]+"_"+line[2].zfill(6)+"_"+line[3].zfill(6)+".mp4"
            video_lbl_num = labels_400[line[0]]
            print(video_path, video_lbl_num)
            writer.writerow([video_path, video_lbl_num])
f_csv.close()
f.close()
```




