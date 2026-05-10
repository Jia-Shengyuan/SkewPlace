import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(ROOT, "results", "experiment_superblue1")
CONFIG = os.path.join(ROOT, "test", "iccad2015.ot", "superblue1.json")


def _run(command, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, check=True, env=merged_env)


def _run_with_log(command, log_path, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(command, check=True, env=merged_env, stdout=f, stderr=subprocess.STDOUT)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    single_skews = [0, 10, 50, 200, 1000]
    dynamic_skews = [0, 10, 50, 200, 1000]
    growing_skews = [200, 1000]
    iterations = 1000

    manifest = {
        "design": "superblue1",
        "config": CONFIG,
        "iterations": iterations,
        "single_skews_ps": single_skews,
        "dynamic_skews_ps": dynamic_skews,
        "growing_skews_ps": growing_skews,
        "results": [],
    }

    no_skew_out = os.path.join(RESULT_DIR, f"no_skew_unified_{iterations}.json")
    _run_with_log([
        sys.executable,
        os.path.join(ROOT, "test", "iccad2015.ot", "run_heterosta_skew_aware_sta_summary.py"),
        CONFIG,
        "0",
        str(iterations),
        "1000",
        "no-skew",
        "0",
        "100",
        no_skew_out,
    ], os.path.join(RESULT_DIR, f"no_skew_unified_{iterations}.log"), env={"HETEROSTA_USE_CUDA": "1"})
    manifest["results"].append({"method": "no-skew", "output": no_skew_out})

    for skew in single_skews:
        out = os.path.join(RESULT_DIR, f"single_skew_{skew}ps_{iterations}.json")
        _run_with_log([
            sys.executable,
            os.path.join(ROOT, "test", "iccad2015.ot", "run_heterosta_skew_aware_sta_summary.py"),
            CONFIG,
            str(skew),
            str(iterations),
            "1000",
            "single-skew",
            "0",
            "100",
            out,
        ], os.path.join(RESULT_DIR, f"single_skew_{skew}ps_{iterations}.log"), env={"HETEROSTA_USE_CUDA": "1"})
        manifest["results"].append({"method": "single-skew", "max_skew_ps": skew, "output": out})

    for skew in dynamic_skews:
        out = os.path.join(RESULT_DIR, f"dynamic_skew_{skew}ps_{iterations}.json")
        _run_with_log([
            sys.executable,
            os.path.join(ROOT, "test", "iccad2015.ot", "run_heterosta_skew_aware_sta_summary.py"),
            CONFIG,
            str(skew),
            str(iterations),
            "1000",
            "dynamic-skew",
            str(skew),
            "100",
            out,
        ], os.path.join(RESULT_DIR, f"dynamic_skew_{skew}ps_{iterations}.log"), env={"HETEROSTA_USE_CUDA": "1"})
        manifest["results"].append({"method": "dynamic-skew", "max_skew_ps": skew, "output": out})

    for skew in growing_skews:
        out = os.path.join(RESULT_DIR, f"growing_skew_{skew}ps_{iterations}.json")
        _run_with_log([
            sys.executable,
            os.path.join(ROOT, "test", "iccad2015.ot", "run_heterosta_skew_aware_sta_summary.py"),
            CONFIG,
            str(skew),
            str(iterations),
            "1000",
            "growing-skew",
            str(skew),
            "100",
            out,
        ], os.path.join(RESULT_DIR, f"growing_skew_{skew}ps_{iterations}.log"), env={"HETEROSTA_USE_CUDA": "1"})
        manifest["results"].append({"method": "growing-skew", "max_skew_ps": skew, "output": out})

    manifest_path = os.path.join(RESULT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({"manifest": manifest_path}, indent=2))


if __name__ == "__main__":
    main()
