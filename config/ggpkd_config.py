import math

from src.criterions.ggpkd_distillation import ROW_COLUMNS, ROW_SETS, ROW_TARGETS
from src.data_utils.batch_samplers import BATCH_SAMPLERS
from src.data_utils.ggpkd_dataset import POOL_SOURCES
from src.ggpkd.graph_builder import (
    HOLDOUT_BANDWIDTHS,
    KNN_MODES,
    NEIGHBOR_SOURCES,
)

from .base_config import BaseConfig


class GGPKDConfig(BaseConfig):
    """GGPKD as defined in story.md §2, with the ablation switches of Tables 3-6.

    Every switch defaults to the method's value, so a run with no GGPKD flags is
    the method.
    """

    distill_method = "ggpkd"

    student_model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "float32"
    student_special_token = "##"
    teacher_special_token = "G"

    # ---- Graph ---------------------------------------------------------------
    # N(j) is the directed top-graph_k of j under `neighbor_source`; targets and
    # bandwidths are always the teacher's. graph_k sets both the neighbourhood and,
    # through tau_j = (s(1) - s(k)) / log k, how sharp each row is.
    graph_k = 100
    neighbor_source = "teacher"  # student: Table 6
    knn_mode = "directed"  # mutual: Table 6
    fixed_bandwidth = False  # one median tau for every row: Table 6
    # Fraction of graph edges withheld from all training terms, for the held-out
    # relations in Table 4. Independent of `seed`, so every arm withholds the same.
    holdout_edge_frac = 0.0
    holdout_seed = 12345
    # Whether tau_j is read off the raw top-k ("full": a target-only holdout, the
    # value story.md §6.4 says must be named as such) or off the edges the holdout
    # left ("surviving": no withheld teacher score reaches training).
    holdout_bandwidth = "full"

    # ---- Objective: L = row_weight * L_row + cal_weight * L_cal -------------------
    # AdamW is nearly scale-invariant, so only the ratio is a real knob (Table 6).
    cal_weight = 0.5
    row_weight = 1.0

    # ---- Ablation switches ---------------------------------------------------
    row_set = "all"  # anchors / non_anchors: Tables 3-4
    row_target = "teacher"  # uniform: Table 3 row 7; shuffled: Table 7
    row_columns = "graph"  # random: Table 3 row 8; pool: Table 7 (dense control)
    # graph: anchors plus their graph rows. random: the same number of texts drawn
    # uniformly, which holds the pool's compute fixed and removes only its
    # structure (Table 7).
    pool_source = "graph"
    row_reweight = False  # 1/p_j row weights: Table 6
    batch_local = False  # pool = the batch, L_cal only: Table 3 rows 3-4
    batch_sampler = "random"  # neighbor: Table 3 rows 2 and 4

    # Which column is the graph node; None picks it from task_type.
    ggpkd_anchor_column = None

    # ---- Training setup ------------------------------------------------------
    batch_size = 64
    epochs = 5
    learning_rate = 3e-5
    min_lr = 3e-6
    num_workers = 4
    eval_every = 0

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    cache_path = "cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt"
    ggpkd_cache_path = "cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/graph.pt"
    ggpkd_log_dir = "logs/ggpkd/qwen3_0_6b_to_minilmv2_h384"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "models/ggpkd/qwen3_0_6b_to_minilmv2_h384/default"
    weights_dir = "models/ggpkd_weights/qwen3_0_6b_to_minilmv2_h384/default"
    final_weights_only = True

    def validate(self):
        """Re-run by main.py after CLI overrides, so every flag combination is checked."""
        for name, value, choices in (
            ("neighbor_source", self.neighbor_source, NEIGHBOR_SOURCES),
            ("knn_mode", self.knn_mode, KNN_MODES),
            ("row_set", self.row_set, ROW_SETS),
            ("row_target", self.row_target, ROW_TARGETS),
            ("row_columns", self.row_columns, ROW_COLUMNS),
            ("batch_sampler", self.batch_sampler, BATCH_SAMPLERS),
            ("pool_source", self.pool_source, POOL_SOURCES),
            ("holdout_bandwidth", self.holdout_bandwidth, HOLDOUT_BANDWIDTHS),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if int(self.graph_k) < 2:
            raise ValueError(f"graph_k must be at least 2, got {self.graph_k}")
        for name in ("cal_weight", "row_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.cal_weight == 0 and self.row_weight == 0:
            raise ValueError(
                "at least one of cal_weight and row_weight must be positive"
            )
        if not 0.0 <= self.holdout_edge_frac < 1.0:
            raise ValueError(
                f"holdout_edge_frac must be in [0, 1), got {self.holdout_edge_frac}"
            )

        row_switches = (
            self.row_set != "all"
            or self.row_target != "teacher"
            or self.row_columns != "graph"
            or self.row_reweight
        )
        if self.batch_local:
            if self.pool_source != "graph":
                raise ValueError(
                    "batch_local already fixes the pool to the batch; pool_source "
                    "has nothing left to choose"
                )
            if self.row_weight > 0 or row_switches:
                raise ValueError(
                    "batch_local has no graph rows: it needs row_weight=0 and the "
                    "default row switches"
                )
            if self.cal_weight <= 0:
                raise ValueError(
                    "batch_local is L_cal over the batch; cal_weight must be > 0"
                )
            if self.batch_size < 2:
                raise ValueError("batch_local needs at least two texts per batch")
        elif row_switches and self.row_weight <= 0:
            raise ValueError(
                "the row switches only change L_row; row_weight must be > 0"
            )
        if self.row_columns != "graph" and self.holdout_edge_frac > 0:
            raise ValueError(
                f"row_columns={self.row_columns!r} scores pairs the graph never "
                "supervised and would draw withheld ones; run it without a holdout"
            )
        if self.holdout_bandwidth == "surviving":
            if self.holdout_edge_frac <= 0:
                raise ValueError(
                    "holdout_bandwidth='surviving' recomputes tau_j on the edges a "
                    "holdout left; it needs holdout_edge_frac > 0"
                )
            if self.fixed_bandwidth:
                raise ValueError(
                    "holdout_bandwidth='surviving' is a per-row bandwidth; it "
                    "cannot be combined with fixed_bandwidth"
                )
        if self.row_reweight and (
            self.row_set == "anchors" or self.batch_sampler != "random"
        ):
            raise ValueError(
                "row_reweight corrects the inclusion probability of uniform anchors; it "
                "needs row_set != 'anchors' and batch_sampler='random'"
            )
