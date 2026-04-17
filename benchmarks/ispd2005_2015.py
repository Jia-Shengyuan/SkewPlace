##
# @file   ispd2005_2015.py
# @author Yibo Lin, Zixuan Jiang, Jiaqi Gu
# @date   Mar 2019
#

import os
import sys
from pyunpack import Archive

if sys.version_info[0] < 3:
    import urllib2 as urllib
else:
    import urllib.request as urllib

baseURL = "http://www.cerc.utexas.edu/~zixuan/"
target_dir = os.path.dirname(os.path.abspath(__file__))

tasks = [
    {
        "archive": "ispd2005dp.tar.xz",
        "marker": os.path.join(target_dir, "ispd2005", "adaptec1", "adaptec1.aux"),
        "dataset_dir": os.path.join(target_dir, "ispd2005"),
    },
    {
        "archive": "ispd2015dp.tar.xz",
        "marker": os.path.join(target_dir, "ispd2015", "mgc_fft_1", "floorplan.def"),
        "dataset_dir": os.path.join(target_dir, "ispd2015"),
    },
]


def ensure_benchmark(task):
    archive_name = task["archive"]
    marker = task["marker"]
    dataset_dir = task["dataset_dir"]
    archive_path = os.path.join(target_dir, archive_name)

    if os.path.exists(marker):
        print("Benchmark already exists, skip: %s" % marker)
        return

    if os.path.isdir(dataset_dir) and os.listdir(dataset_dir):
        print("Benchmark directory already exists, skip download: %s" % dataset_dir)
        return

    if os.path.exists(archive_path):
        print("Use local archive %s" % archive_path)
    else:
        file_url = baseURL + archive_name
        print("Download from %s to %s" % (file_url, archive_path))
        response = urllib.urlopen(file_url)
        content = response.read()
        with open(archive_path, "wb") as f:
            f.write(content)

    print("Uncompress %s to %s" % (archive_path, target_dir))
    Archive(archive_path).extractall(target_dir)

    if os.path.exists(marker):
        print("Benchmark ready: %s" % marker)
    else:
        raise RuntimeError("Extraction finished but marker is missing: %s" % marker)


for t in tasks:
    ensure_benchmark(t)
