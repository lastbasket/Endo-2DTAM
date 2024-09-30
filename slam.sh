export OUTPUT_NAME=C3VD_base
export DOWN_SCALE=2
for i in {0..9}
do
    echo $i
    export SCENE_NUM=$i
    python scripts/main.py configs/c3vd/c3vd_base.py
    python scripts/calc_metrics.py --gt data/C3VD --render experiments/${OUTPUT_NAME} --test_single
done
