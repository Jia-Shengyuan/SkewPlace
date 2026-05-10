import json
import os
import re
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_with_log(command, log_path, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    begin = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(command, check=True, env=merged_env, stdout=f, stderr=subprocess.STDOUT)
    return time.time() - begin


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_result(manifest, result):
    manifest.setdefault("results", []).append(result)


def _abs_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _clock_period_ps(config_path):
    config = _load_json(config_path)
    sdc_path = _abs_path(ROOT, config.get("sdc_input"))
    pattern = re.compile(r"create_clock\b.*?-period\s+([0-9eE+\-.]+)")
    with open(sdc_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


def _skew_tag(value):
    text = ("%.6f" % float(value)).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _scale_skews(reference_skews, clock_period, reference_period=9500.0):
    if not clock_period:
        return [float(skew) for skew in reference_skews]
    scale = float(clock_period) / float(reference_period)
    return [float(skew) * scale for skew in reference_skews]


def _extract_field(data, path):
    current = data
    for item in path:
        if not isinstance(current, dict):
            return None
        current = current.get(item)
    return current


def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def _write_markdown(manifest, markdown_path):
    rows = []
    for result in manifest.get("results", []):
        output = result.get("output")
        data = None
        if output and os.path.exists(output):
            try:
                data = _load_json(output)
            except Exception:
                data = None
        data = data or {}
        rows.append({
            "design": result.get("design", manifest.get("design")),
            "clock_period_ps": data.get("clock_period_ps", result.get("clock_period_ps")),
            "mode": result.get("mode"),
            "max_skew_ps": result.get("max_skew_ps"),
            "raw_wns_ps": _extract_field(data, ["raw_sta", "setup", "worst_slack_ps"]),
            "raw_tns_ps": _extract_field(data, ["raw_sta", "setup", "tns_ps"]),
            "skewed_wns_ps": _extract_field(data, ["skew_aware_sta", "setup", "worst_slack_ps"]),
            "skewed_tns_ps": _extract_field(data, ["skew_aware_sta", "setup", "tns_ps"]),
            "hpwl": _extract_field(data, ["placement_metrics", "hpwl"]),
            "path_count_selected": data.get("path_count_selected"),
            "runtime_sec": result.get("wall_runtime_sec", data.get("runtime_sec")),
            "status": "error" if result.get("error") or data.get("stopped_due_to_error") else "ok",
            "output": output,
        })

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# Skew Experiment Sweep\n\n")
        f.write("本表使用 HeteroSTA post-place timing，placement 阶段分别跑 no-skew、dynamic-skew、growing-skew。\n\n")
        f.write("说明：raw WNS/TNS 来自 HeteroSTA full-design setup report；skewed WNS/TNS 来自递增 top-N 到收敛后的 reg-to-reg path 集合上的外部 skew-aware 重算。\n\n")
        f.write("| Benchmark | Clock (ps) | Mode | Max skew (ps) | Raw WNS (ps) | Raw TNS (ps) | Skewed WNS (ps) | Skewed TNS (ps) | HPWL | Paths | Runtime (s) | Status |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {design} | {clock_period_ps} | {mode} | {max_skew_ps} | {raw_wns_ps} | {raw_tns_ps} | {skewed_wns_ps} | {skewed_tns_ps} | {hpwl} | {path_count_selected} | {runtime_sec} | {status} |\n".format(
                    design=row["design"],
                    clock_period_ps=_fmt(row["clock_period_ps"]),
                    mode=row["mode"],
                    max_skew_ps=_fmt(row["max_skew_ps"]),
                    raw_wns_ps=_fmt(row["raw_wns_ps"]),
                    raw_tns_ps=_fmt(row["raw_tns_ps"]),
                    skewed_wns_ps=_fmt(row["skewed_wns_ps"]),
                    skewed_tns_ps=_fmt(row["skewed_tns_ps"]),
                    hpwl=_fmt(row["hpwl"]),
                    path_count_selected=_fmt(row["path_count_selected"]),
                    runtime_sec=_fmt(row["runtime_sec"]),
                    status=row["status"],
                )
            )
        f.write("\n")
        f.write("## Outputs\n\n")
        for row in rows:
            f.write("- `{}`\n".format(row["output"]))


def _write_manifest(manifest, manifest_path, markdown_path):
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    _write_markdown(manifest, markdown_path)


def _result_key(result):
    return (
        result.get("design"),
        result.get("mode"),
        round(float(result.get("max_skew_ps", 0.0)), 6),
    )


def _load_existing_results(manifest_path):
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    existing = {}
    for result in manifest.get("results", []):
        existing[_result_key(result)] = result
    return existing


def _json_completed(path):
    if not path or not os.path.exists(path):
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False
    return not data.get("stopped_due_to_error")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python run_skew_experiment_suite.py <result_dir> <config_jsons_csv> [reference_skews_csv] [iterations] [path_count] [skew_n] [reference_period_ps]"
        )

    result_dir = sys.argv[1]
    configs = [item for item in sys.argv[2].split(",") if item.strip()]
    reference_skews = [float(item) for item in sys.argv[3].split(",") if item.strip()] if len(sys.argv) >= 4 else [1.0, 100.0, 1000.0, 9000.0]
    iterations = int(sys.argv[4]) if len(sys.argv) >= 5 else 1000
    path_count = int(sys.argv[5]) if len(sys.argv) >= 6 else 1000
    skew_n = int(sys.argv[6]) if len(sys.argv) >= 7 else 100
    reference_period = float(sys.argv[7]) if len(sys.argv) >= 8 else 9500.0

    os.makedirs(result_dir, exist_ok=True)
    summary_script = os.path.join(ROOT, "test", "iccad2015.ot", "run_heterosta_skew_aware_sta_summary.py")
    manifest_path = os.path.join(result_dir, "skew_sweep_manifest.json")
    markdown_path = os.path.join(result_dir, "skew_sweep_results.md")

    manifest = {
        "configs": configs,
        "iterations": iterations,
        "path_count": path_count,
        "skew_n": skew_n,
        "reference_period_ps": reference_period,
        "reference_skews_ps": reference_skews,
        "results": [],
    }
    existing_results = _load_existing_results(manifest_path)

    env = {"HETEROSTA_USE_CUDA": "1"}

    for config in configs:
        config = _abs_path(ROOT, config)
        config_name = os.path.splitext(os.path.basename(config))[0]
        clock_period = _clock_period_ps(config)
        skews = _scale_skews(reference_skews, clock_period, reference_period=reference_period)
        runs = [("no-skew", 0.0)]
        for mode in ("dynamic-skew", "growing-skew"):
            runs.extend((mode, skew) for skew in skews)

        for mode, skew in runs:
            skew_tag = _skew_tag(skew)
            output = os.path.join(result_dir, f"{config_name}_{mode}_{skew_tag}ps_summary.json")
            log = os.path.join(result_dir, f"{config_name}_{mode}_{skew_tag}ps_summary.log")
            result = {
                "kind": "summary",
                "design": config_name,
                "config": config,
                "clock_period_ps": clock_period,
                "mode": mode,
                "max_skew_ps": skew,
                "output": output,
                "log": log,
            }
            existing = existing_results.get(_result_key(result))
            if existing and not existing.get("error") and _json_completed(existing.get("output")):
                _append_result(manifest, existing)
                _write_manifest(manifest, manifest_path, markdown_path)
                continue
            try:
                runtime = _run_with_log([
                    sys.executable,
                    summary_script,
                    config,
                    str(skew),
                    str(iterations),
                    str(path_count),
                    mode,
                    str(skew),
                    str(skew_n),
                    output,
                ], log, env=env)
                result["wall_runtime_sec"] = runtime
            except subprocess.CalledProcessError as exc:
                result["error"] = {
                    "type": type(exc).__name__,
                    "returncode": exc.returncode,
                    "message": str(exc),
                }
            _append_result(manifest, result)
            _write_manifest(manifest, manifest_path, markdown_path)

    _write_manifest(manifest, manifest_path, markdown_path)
    print(json.dumps({"manifest": manifest_path, "markdown": markdown_path}, indent=2))


if __name__ == "__main__":
    main()
