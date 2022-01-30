# cd /path/to/this/repo
export POINTNAV_VO_ROOT="/Extra/lwy/habitat/PointNav-VO/"

export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
# conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type rl \
--noise 1 \
--run-type eval \
--addr 127.0.1.1 \
--port 8338