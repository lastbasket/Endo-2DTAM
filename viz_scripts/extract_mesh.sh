export OUTPUT_NAME=Base_final
export DOWN_SCALE=2
for i in {0..9}
do
    echo $i
    export SCENE_NUM=$i
    python scripts/extract_mesh.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single --data_config configs/data/c3vd.yaml
done