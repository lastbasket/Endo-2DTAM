export OUTPUT_NAME=Base_final
export DOWN_SCALE=2
export SCENE_NUM=0
export JSON_NAME='ScreenCamera_2024-09-19-22-51-30'

python viz_scripts/extract_video.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml
python viz_scripts/extract_video.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml --gray
python viz_scripts/extract_video_gt.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml
python viz_scripts/extract_video_gt.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml --gray

python viz_scripts/make_video.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml