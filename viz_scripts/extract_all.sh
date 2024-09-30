export OUTPUT_NAME=Base_final
export DOWN_SCALE=2
export SCENE_NUM=9

python viz_scripts/extract_all.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml