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

from dreamplace import Params
from dreamplace import PlaceDB


def _abs_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def probe(config_path):
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

    placedb = PlaceDB.PlaceDB()
    placedb(params)
    return {
        "design": params.design_name(),
        "num_nodes": int(placedb.num_nodes),
        "num_movable_nodes": int(placedb.num_movable_nodes),
        "num_terminals": int(placedb.num_terminals),
        "num_nets": int(len(placedb.net_names)),
        "num_pins": int(len(placedb.pin2net_map)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)-7s] %(name)s - %(message)s", stream=sys.stdout)
    if len(sys.argv) < 2:
        raise SystemExit("usage: python probe_benchmark_sizes.py <json> [<json> ...]")
    rows = [probe(path) for path in sys.argv[1:]]
    print(json.dumps(rows, indent=2))
