import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSTALL = os.path.join(ROOT, "install")
INSTALL_DREAMPLACE = os.path.join(INSTALL, "dreamplace")
if INSTALL not in sys.path:
    sys.path.insert(0, INSTALL)
if INSTALL_DREAMPLACE not in sys.path:
    sys.path.insert(0, INSTALL_DREAMPLACE)

from dreamplace import NonLinearPlace
from dreamplace import Params
from dreamplace import PlaceDB
from dreamplace import Timer


def _abs_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def run(config_path, iterations=1000, checkpoints=None, placement_mode="no-skew", placement_max_skew=0.0, skew_n=100):
    if checkpoints is None:
        checkpoints = [500]

    params = Params.Params()
    params.load(config_path)

    params.lef_input = _abs_path(ROOT, params.lef_input)
    params.def_input = _abs_path(ROOT, params.def_input)
    params.verilog_input = _abs_path(ROOT, params.verilog_input)
    params.early_lib_input = _abs_path(ROOT, params.early_lib_input)
    params.late_lib_input = _abs_path(ROOT, params.late_lib_input)
    params.sdc_input = _abs_path(ROOT, params.sdc_input)
    params.result_dir = os.path.join(ROOT, "results")
    params.legalize_flag = 0
    params.detailed_place_flag = 0
    params.global_place_stages[0]["iteration"] = iterations
    params.dump_global_place_checkpoint_steps = checkpoints
    params.timer_engine = "heterosta"
    params.heterosta_use_cuda = int(os.environ.get("HETEROSTA_USE_CUDA", "1"))
    params.useful_skew_weighting_flag = 1 if placement_mode in ("dynamic-skew", "growing-skew") else 0
    params.useful_skew_weighting_n = int(skew_n)
    params.useful_skew_schedule = "growing" if placement_mode == "growing-skew" else "fixed"
    params.useful_skew_max_skew = float(placement_max_skew)

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine="heterosta")
    timer(params, placedb)

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer(params, placedb, learning_rate_value)


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )

    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python run_heterosta_gp_checkpoints.py <json> [iterations] [checkpoints] [placement_mode] [placement_max_skew_ps] [skew_n]"
        )

    config_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    checkpoints = [int(item) for item in sys.argv[3].split(",") if item] if len(sys.argv) >= 4 else [500]
    placement_mode = sys.argv[4] if len(sys.argv) >= 5 else "no-skew"
    placement_max_skew = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.0
    skew_n = int(sys.argv[6]) if len(sys.argv) >= 7 else 100

    run(
        config_path,
        iterations=iterations,
        checkpoints=checkpoints,
        placement_mode=placement_mode,
        placement_max_skew=placement_max_skew,
        skew_n=skew_n,
    )
