1.������.npy�ļ���Generating detections��  ����֮ǰӦ�ñ���ü�����ļ�
python tools/generate_detections.py --model=../resources/networks/mars-small128.pb  --mot_dir=../MOT16/train  --output_dir=../resources/detections/MOT16_train
2.���и��ٳ���Running the tracker��
python deep_sort_app.py  --sequence_dir=./MOT16/train/try1     --detection_file=./resources/detections/MOT16_train/try1.npy     --min_confidence=0.5    --nn_budget=100     --display=True
--sequence_dir=E:/DeeCamp/deecamp/deep_sort-master/MOT16/train/try3,E:/DeeCamp/deecamp/deep_sort-master/MOT16/train/try4,E:/DeeCamp/deecamp/deep_sort-master/MOT16/train/try1     --detection_file=E:/DeeCamp/deecamp/deep_sort-master/resources/detections/MOT16_train/try3all.npy,E:/DeeCamp/deecamp/deep_sort-master/resources/detections/MOT16_train/try4all.npy,E:/DeeCamp/deecamp/deep_sort-master/resources/detections/MOT16_train/try1all.npy    --min_confidence=0.5    --nn_budget=100     --display=True

3.����������Ƶ show_results.py  
--sequence_dir=./MOT16/train/try1  --result_file=./hypothesis1.txt  --detection_file=./resources/detections/MOT16_train/try1new.npy     --output_file=./c1.mp4
--sequence_dir=./MOT16/train/try3  --result_file=./hypothesis3.txt  --detection_file=./resources/detections/MOT16_train/try3new.npy     --output_file=./c3.mp4
