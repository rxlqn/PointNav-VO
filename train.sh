# cd ${POINTNAV_VO_ROOT}
export POINTNAV_VO_ROOT="/Extra/lwy/habitat/PointNav-VO/"

export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
# conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 4 \
--task-type rl \
--noise 1 \
--run-type train \
--addr 127.0.1.1 \
--port 8339
# num_env = 4 一个进程同时几个环境
## 现在训练集用的val_mini所以最多并行环境只有3个，现在用的是2个