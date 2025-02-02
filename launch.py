import os
import datetime
import argparse


DEFAULT_ADDR = "127.0.1.1"
DEFAULT_PORT = "8338"

CMD_RL = "export CUDA_LAUNCH_BLOCKING=1 && \
       export PYTHONPATH={}:$PYTHONPATH && \
       python -u -m torch.distributed.launch \
       --nproc_per_node={} \
       --master_addr {} \
       --master_port {} \
       --use_env \
       {} \
       --task-type {} \
       --noise {} \
       --exp-config {} \
       --run-type {} \
       --n-gpu {} \
       --cur-time {}"

CMD_VO = "export CUDA_LAUNCH_BLOCKING=1 && \
       export PYTHONPATH={}:$PYTHONPATH && \
       python {} \
       --task-type {} \
       --noise {} \
       --exp-config {} \
       --run-type {} \
       --n-gpu {} \
       --cur-time {}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        choices=["rl", "vo"],
        required=False,
        help="Specify the category of the task",
        default="rl"
    )
    parser.add_argument(
        "--noise",
        type=int,
        required=False,
        help="Whether adding noise into environment",
        default=1
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=False,
        help="run type of the experiment (train or eval)",
        default="train"
    )
    parser.add_argument(
        "--repo-path", type=str, required=False, help="path to PointNav repo", default="/Extra/lwy/habitat/PointNav-VO/"
    )
    parser.add_argument(
        "--n_gpus", type=int, required=False, help="path to PointNav repo", default=1
    )
    parser.add_argument(
        "--addr", type=str, required=False, help="master address", default = DEFAULT_ADDR
    )
    parser.add_argument(
        "--port", type=str, required=False, help="master port", default = DEFAULT_PORT
    )

    args = parser.parse_args()

    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

    if args.task_type == "rl":
        cur_config_f = os.path.join(args.repo_path, "configs/rl/ddppo_pointnav.yaml")
    elif args.task_type == "vo":
        cur_config_f = os.path.join(args.repo_path, "configs/vo/vo_pointnav.yaml")
    else:
        pass

    if "rl" in args.task_type:
        tmp_cmd = CMD_RL.format(
            args.repo_path,
            args.n_gpus,
            args.addr,
            args.port,
            # {}/point_nav/run.py
            os.path.join(args.repo_path, "pointnav_vo/run.py"),
            args.task_type,
            args.noise,
            cur_config_f,
            args.run_type,
            args.n_gpus,
            cur_time,
        )
    elif "vo" in args.task_type:
        tmp_cmd = CMD_VO.format(
            args.repo_path,
            os.path.join(args.repo_path, "pointnav_vo/run.py"),
            args.task_type,
            args.noise,
            cur_config_f,
            args.run_type,
            args.n_gpus,
            cur_time,
        )
    else:
        raise ValueError

    print("\n", tmp_cmd, "\n")

    os.system(tmp_cmd)
