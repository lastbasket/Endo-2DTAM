export OUTPUT_NAME=C3VD_base
export DOWN_SCALE=2
for i in {2..2}
do
    echo $i
    export SCENE_NUM=$i
    python scripts/calc_metrics.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single
done