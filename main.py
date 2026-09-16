import argparse
import sys

from config import TALAS_PAPER_PAIRS, get_talas_paper_pair
from distiller import KnowledgeDistiller
from src.methods import METHOD_NAMES, get_method


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Embeddings Model"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="cdm",
        choices=list(METHOD_NAMES),
        help="Distillation method to use",
    )

    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to training data CSV file"
    )

    parser.add_argument(
        "--student_model", type=str, default=None, help="Student model name or path"
    )
    parser.add_argument(
        "--teacher_model", type=str, default=None, help="Teacher model name or path"
    )
    parser.add_argument(
        "--talas_pair",
        choices=sorted(TALAS_PAPER_PAIRS),
        default=None,
        help="Canonical TALAS teacher-student pair from the paper",
    )

    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Evaluate every N epochs; 0 disables per-epoch evaluation",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum sequence length"
    )

    # GGPKD. Every flag defaults to the method (story.md §2); the ablation
    # switches are the arms of story.md Tables 3-6.
    parser.add_argument("--graph_k", type=int, default=None)
    parser.add_argument(
        "--neighbor_source", choices=["teacher", "student"], default=None
    )
    parser.add_argument("--knn_mode", choices=["directed", "mutual"], default=None)
    parser.add_argument(
        "--fixed_bandwidth",
        action="store_true",
        help="One median bandwidth for every row instead of the per-row tau_j",
    )
    parser.add_argument(
        "--holdout_edge_frac",
        type=float,
        default=None,
        help="Fraction of graph edges withheld from every training term (Table 4)",
    )
    parser.add_argument("--holdout_seed", type=int, default=None)
    parser.add_argument(
        "--holdout_bandwidth",
        choices=["full", "surviving"],
        default=None,
        help="Whether tau_j is read off the raw top-k (target-only holdout) or off "
        "the edges the holdout left",
    )
    parser.add_argument(
        "--cal_weight", type=float, default=None, help="Weight of L_cal"
    )
    parser.add_argument(
        "--row_weight", type=float, default=None, help="Weight of L_row"
    )
    parser.add_argument(
        "--row_set",
        choices=["all", "anchors", "non_anchors"],
        default=None,
        help="Which pool texts are supervised rows",
    )
    parser.add_argument(
        "--row_target",
        choices=["teacher", "uniform", "shuffled"],
        default=None,
        help="Row target values: teacher softmax (method), uniform on the same "
        "columns, or the teacher's own values permuted among them",
    )
    parser.add_argument(
        "--row_columns",
        choices=["graph", "random", "pool"],
        default=None,
        help="Row columns: graph neighbours in the pool (method), random pool texts, "
        "or every other pool text (dense relational KD on the same pool)",
    )
    parser.add_argument(
        "--row_reweight",
        action="store_true",
        help="Weight each row by 1/P(text in pool) (GraphSAINT normalization)",
    )
    parser.add_argument(
        "--pool_source",
        choices=["graph", "random"],
        default=None,
        help="Pool texts: the anchors' graph rows (method) or the same number of "
        "uniformly drawn corpus texts",
    )
    parser.add_argument(
        "--batch_local",
        action="store_true",
        help="In-batch baseline: the pool is the batch and the loss is L_cal only",
    )
    parser.add_argument(
        "--batch_sampler",
        choices=["random", "neighbor"],
        default=None,
        help="Batch composition: i.i.d. (method) or one graph neighbourhood per batch",
    )
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--ggpkd_cache_path", type=str, default=None)
    parser.add_argument("--ggpkd_log_dir", type=str, default=None)
    parser.add_argument(
        "--pooling_method",
        choices=["last_token", "mean", "cls"],
        default=None,
    )

    parser.add_argument("--w_task", type=float, default=None, help="Task loss weight")
    parser.add_argument(
        "--alpha_dtw", type=float, default=None, help="DTW KD loss weight"
    )
    parser.add_argument(
        "--rkd_distance_weight",
        type=float,
        default=None,
        help="RKD distance-wise loss weight",
    )
    parser.add_argument(
        "--rkd_angle_weight",
        type=float,
        default=None,
        help="RKD angle-wise loss weight",
    )
    parser.add_argument(
        "--task_type",
        choices=["single_cls", "pair_cls", "pair_reg"],
        default=None,
        help="Training task contract",
    )

    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Optional durable directory for per-epoch student weights",
    )
    parser.add_argument(
        "--final_weights_only",
        action="store_true",
        help="Save only the final student state dict, not training checkpoints",
    )
    parser.add_argument(
        "--prepare_cache_only",
        action="store_true",
        help="Prepare and validate a cached-teacher method, then exit before training",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of dataloader workers"
    )

    return parser.parse_args()


def get_config(method: str, args):
    # The registry is the single list of methods: `--method` choices, the config
    # class, and the capabilities the distiller reads all come from one entry.
    spec = get_method(method)
    config = spec.config_cls()

    if args.talas_pair is not None:
        if method != "talas":
            raise ValueError("--talas_pair is supported only with --method talas")
        preset = get_talas_paper_pair(args.talas_pair)
        config.paper_pair = args.talas_pair
        config.teacher_model_name = preset["teacher"]
        config.student_model_name = preset["student"]
        config.pooling_method = preset["pooling_method"]
        # `TALASConfig.cache_path` is an f-string interpolated once at class
        # definition time against DEFAULT_TALAS_PAIR, so assigning `paper_pair`
        # above does not move it. Without this line `--talas_pair X` reads the
        # default pair's teacher cache, and nothing downstream would notice: the
        # cache-hit gate gets no further than os.path.exists, and the validator
        # checks shape/dtype/finiteness but not teacher identity. Two of the paper
        # pairs have 1024-dim teachers, so it fails silently rather than loudly.
        config.cache_path = f"cache/talas/{args.talas_pair}/teacher_train.pt"

    if args.train_data is not None:
        config.train_data_path = args.train_data

    if args.student_model is not None:
        config.student_model_name = args.student_model
    if args.teacher_model is not None:
        config.teacher_model_name = args.teacher_model

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.eval_every is not None:
        if args.eval_every < 0:
            raise ValueError("--eval_every must be non-negative")
        config.eval_every = args.eval_every
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.max_length is not None:
        config.max_length = args.max_length

    ggpkd_overrides = (
        "graph_k",
        "neighbor_source",
        "knn_mode",
        "holdout_edge_frac",
        "holdout_seed",
        "holdout_bandwidth",
        "pool_source",
        "cal_weight",
        "row_weight",
        "row_set",
        "row_target",
        "row_columns",
        "batch_sampler",
        "cache_path",
        "ggpkd_cache_path",
        "ggpkd_log_dir",
        "pooling_method",
    )
    for name in ggpkd_overrides:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    for name in ("fixed_bandwidth", "row_reweight"):
        if getattr(args, name):
            setattr(config, name, True)
    if args.batch_local:
        config.batch_local = True
        # The baseline has no graph rows, so L_row is off unless asked for (and
        # then validate() refuses it).
        if args.row_weight is None:
            config.row_weight = 0.0

    if args.w_task is not None:
        config.w_task = args.w_task
    if args.alpha_dtw is not None:
        config.alpha_dtw = args.alpha_dtw
    if args.rkd_distance_weight is not None:
        config.rkd_distance_weight = args.rkd_distance_weight
    if args.rkd_angle_weight is not None:
        config.rkd_angle_weight = args.rkd_angle_weight
    if args.task_type is not None:
        config.task_type = args.task_type

    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.weights_dir is not None:
        config.weights_dir = args.weights_dir
    if args.final_weights_only:
        config.final_weights_only = True

    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug_align = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # The CLI writes attributes onto an already-constructed config, so the
    # constructor's checks have long since run. Re-run them over the final values.
    config.validate()

    return config


def main():
    args = parse_args()

    if args.prepare_cache_only and not get_method(args.method).uses_teacher_cache:
        cached = ", ".join(
            name for name in METHOD_NAMES if get_method(name).uses_teacher_cache
        )
        raise SystemExit(
            f"--prepare_cache_only needs a method with a teacher cache: {cached}"
        )

    config = get_config(args.method, args)

    print("\n" + "=" * 70)
    print(f"Configuration for {args.method.upper()} method:")
    print("=" * 70)
    for k, v in config.to_dict().items():
        print(f"  {k:25s} : {v}")
    print("=" * 70 + "\n")

    try:
        distiller = KnowledgeDistiller(config)
    except Exception as e:
        print(f"Error creating distiller: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if args.prepare_cache_only:
        print(
            f"{args.method.upper()} teacher cache prepared and validated: {config.cache_path}"
        )
        return

    try:
        distiller.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
