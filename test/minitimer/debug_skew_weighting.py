import json
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


def _net_snapshot(placedb, limit=20):
    rows = []
    for net_id, name in enumerate(placedb.net_names[:limit]):
        rows.append({
            "net_id": int(net_id),
            "net_name": name.decode("utf-8") if isinstance(name, bytes) else str(name),
            "weight": float(placedb.net_weights[net_id]),
            "criticality": float(placedb.net_criticality[net_id]),
        })
    return rows


def run(config_path, use_skew=False):
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
    params.global_place_stages[0]["iteration"] = 10
    params.useful_skew_weighting_flag = 1 if use_skew else 0
    params.useful_skew_weighting_n = 10
    params.useful_skew_max_skew = 20.0

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer(params, placedb, learning_rate_value)

    timing_op = placer.op_collections.timing_op
    pos = placer.pos[0].data.clone().cpu()
    timing_op(pos)
    timing_op.timer.update_timing()

    before = _net_snapshot(placedb)
    timing_op.update_net_weights(max_net_weight=placedb.max_net_weight, n=10)
    after = _net_snapshot(placedb)

    summary = {
        "use_skew": use_skew,
        "before": before,
        "after": after,
        "weighting_stats": getattr(timing_op, "last_useful_skew_weighting_stats", {}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) < 2:
        raise SystemExit("usage: python debug_skew_weighting.py <json> [use_skew]")
    config_path = sys.argv[1]
    use_skew = bool(int(sys.argv[2])) if len(sys.argv) >= 3 else False
    run(config_path, use_skew=use_skew)
