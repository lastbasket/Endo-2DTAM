export OUTPUT_NAME=Base_final
export DOWN_SCALE=2
export SCENE_NUM=0

python viz_scripts/extract_video.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml
python viz_scripts/extract_video.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml --gray