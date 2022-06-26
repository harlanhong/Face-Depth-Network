
# Train
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 2  --heigh 256 --width 256 --dataset vox  --sample_num 100000 --model_name taking_head_10w --data_path vox2

#Test
CUDA_VISIBLE_DEVICES=1 python produce_depth_video.py --model_name tmp/celeb2_10w/models/weights_19 --video_path 7PbDDjXgYzU
