import json
import logging
import os
import sys

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
from test.iccad2015.ot.run_useful_skew_summary import _abs_path
from test.iccad2015.ot.run_useful_skew_summary import _baseline_metrics
from test.iccad2015.ot.run_useful_skew_summary import _useful_skew_metrics


def run(config_path, checkpoint_path, baseline_n=100, max_skew=50.0, sweep_ns=None):
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

    if sweep_ns is None:
        sweep_ns = [baseline_n]

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer.load(params, placedb, checkpoint_path)

    timing_op = placer.op_collections.timing_op
    pos = placer.pos[0].data.clone().cpu()
    timing_op(pos)
    timing_op.timer.update_timing()

    baseline = _baseline_metrics(timing_op.timer, baseline_n)
    useful_skew_sweep = [
        _useful_skew_metrics(timing_op.timer, item_n, max_skew)
        for item_n in sweep_ns
    ]

    summary = {
        "design": params.design_name(),
        "checkpoint_path": checkpoint_path,
        "baseline": baseline,
        "useful_skew_sweep": useful_skew_sweep,
        "max_skew": max_skew,
    }

    out_dir = os.path.join(ROOT, "results", "iccad2015.ot")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_tag = os.path.basename(checkpoint_path).replace(".pklz", "")
    out_path = os.path.join(out_dir, checkpoint_tag + "_useful_skew.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(json.dumps({"summary_path": out_path}))


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python evaluate_useful_skew_checkpoint.py <json> <checkpoint.pklz> [baseline_n] [max_skew] [sweep_ns]"
        )

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    baseline_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    max_skew = float(sys.argv[4]) if len(sys.argv) >= 5 else 50.0
    sweep_ns = [int(item) for item in sys.argv[5].split(",") if item] if len(sys.argv) >= 6 else None

    run(config_path, checkpoint_path, baseline_n=baseline_n, max_skew=max_skew, sweep_ns=sweep_ns)
