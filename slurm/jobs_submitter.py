from idr_pytools import gpu_jobs_submitter, display_slurm_queue, search_log
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Argument Parser for Job Submission"
    )

    # Arguments
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        metavar="CONFIG",
        help="Configs yaml file",
    )

    parser.add_argument(
        "--n_gpu",
        type=int,
        nargs="+",
        default=[4],
        metavar="N",
        help="Number of GPUs to reserve for a job. Default is 1 GPU. Maximum is 512 GPUs. Can also provide a list of GPU numbers.",
    )
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        default="pytorch-gpu/py3/2.0.1",
        metavar="MODULE",
        help="Name of the module to load. Only one module name allowed.",
    )
    parser.add_argument(
        "--singularity",
        type=str,
        required=True,
        metavar="SIF_IMAGE",
        help="Name of the SIF image to load. idrcontmgr command should have been applied beforehand. See documentation for Singularity container usage.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ddpm",
        metavar="JOB_NAME",
        help="Name of the job. It will be displayed in SLURM queue and included in log names. Default is the name of the Python script specified in srun_commands.",
    )

    parser.add_argument(
        "--n_gpu_per_task",
        type=int,
        metavar="N",
        help="Number of GPUs associated with a task. Default is 1 GPU / task.",
    )

    parser.add_argument(
        "--time_max",
        type=str,
        default="01:55:00",
        metavar="HH:MM:SS",
        help="Maximum duration of the job. Default is '02:00:00'.",
    )

    parser.add_argument(
        "--qos",
        type=str,
        default="qos_gpu-dev",
        metavar="QOS",
        help="QoS to use if different from default 'qos_gpu-t3'. Default is 'qos_gpu-t4', 'qos_gpu-dev'.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="gpu_p13",
        metavar="PARTITION",
        help="Partition to use if different from default 'gpu_p13'. Default is 'gpu_p2', 'gpu_p2l', 'gpu_p2s'.",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="v100-32g",
        metavar="CONSTRAINT",
        help="Constraint to use. 'v100-32g' or 'v100-16g'. This forces the use of either 32GB or 16GB GPUs.",
    )

    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=8,
        metavar="N",
        help="Number of CPUs to associate with each task. Default is 10 for default partition or 3 for gpu_p2 partition.",
    )

    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Force the use of a node exclusively.",
    )
    parser.add_argument(
        "--account",
        type=str,
        required=True,
        metavar="ACCOUNT",
        help="Allocation of GPU hours to use. Mandatory if multiple hour allocations and/or projects are accessible.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode. Default is 0. Value 1 adds NVIDIA debugging traces to logs.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    command = [
        f"python -m torch.distributed.run --standalone --nproc_per_node gpu main.py --yaml_path {config}"
        for config in args.configs
    ]

    gpu_jobs_submitter(
        command,
        n_gpu=args.n_gpu,
        module=args.module,
        singularity=args.singularity,
        name=args.name,
        # n_gpu_per_task=args.n_gpu_per_task,
        time_max=args.time_max,
        qos=args.qos,
        partition=args.partition,
        constraint=args.constraint,
        cpus_per_task=args.cpus_per_task,
        exclusive=args.exclusive,
        account=args.account,
        verbose=args.verbose,
    )
