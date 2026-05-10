import json
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
from dreamplace import Timer


def _abs_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _decode_pin_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _normalize_pin_name(name):
    if ":" in name and "/" not in name:
        return name.replace(":", "/")
    return name


def run(config_path, output_json, output_csv):
    params = Params.Params()
    params.load(config_path)

    params.lef_input = _abs_path(ROOT, params.lef_input)
    params.def_input = _abs_path(ROOT, params.def_input)
    params.verilog_input = _abs_path(ROOT, params.verilog_input)
    params.early_lib_input = _abs_path(ROOT, params.early_lib_input)
    params.late_lib_input = _abs_path(ROOT, params.late_lib_input)
    params.sdc_input = _abs_path(ROOT, params.sdc_input)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    raw = timer.raw_timer

    raw.debug_dump_netlist_pin_names(output_csv)

    indices = [0, 1, 2, 10, 100, len(placedb.pin_names) // 2, len(placedb.pin_names) - 1]
    checks = []
    for idx in indices:
        raw_pin_name = _decode_pin_name(placedb.pin_names[idx])
        normalized_pin_name = _normalize_pin_name(raw_pin_name)
        heterosta_idx = int(raw.lookup_pin(normalized_pin_name))
        checks.append(
            {
                "dreamplace_idx": int(idx),
                "pin_name": raw_pin_name,
                "normalized_pin_name": normalized_pin_name,
                "heterosta_lookup": heterosta_idx,
                "index_match": heterosta_idx == int(idx),
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
        f.write("\n")

    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python debug_pin_order_check.py <json> <output_json> <output_csv>"
        )

    run(sys.argv[1], sys.argv[2], sys.argv[3])
