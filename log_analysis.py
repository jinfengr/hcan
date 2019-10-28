from optparse import OptionParser
from os import listdir
from os.path import isfile, join
import numpy
from optparse import OptionParser


def main(filters, monitor):
    def filter_file(file, filters):
        if ".log" not in file:
            return True
        for filter in filters.split():
            if filter not in file:
                return True
        return False

    path = "emnlp_logs"
    monitors = {}
    for file in listdir(path):
        if filter_file(file, filters):
            continue
        # print(file)
        monitor_val = file.split(monitor)[1].split("_")[0]
        assert len(monitor_val) != 0
        if monitor_val not in monitors:
            monitors[monitor_val] = {'MAP': [], 'MRR':[], 'P30': []}
        with open(join(path, file)) as f:
            last_line = f.readlines()[-1]
            if not last_line.startswith("MAP"):
                continue
            groups = last_line.split()
            map, p30, mrr = float(groups[1]), float(groups[3]), float(groups[5])
            monitors[monitor_val]['MAP'].append(map)
            monitors[monitor_val]['MRR'].append(mrr)
            monitors[monitor_val]['P30'].append(p30)

    print(monitors.keys())
    for monitor_val in monitors:
        print("-----------", monitor_val, "-----------")
        for metric in ['MAP', 'MRR', 'P30']:
            vals = monitors[monitor_val][metric]
            print(metric, len(vals), max(vals), min(vals), sum(vals)/len(vals))
            print("10perc: %.4f, 50perc: %.4f, 90perc: %.4f" % (
                numpy.percentile(vals, 10),
                numpy.percentile(vals, 50),
                numpy.percentile(vals, 90)))
            print(sorted(vals))
            print("\n")


def create_option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--filters", action="store", type=str, dest="filters")
    parser.add_option("-m", "--monitor", action="store", type=str, dest="monitor")
    return parser


if __name__ == "__main__":
    parser = create_option_parser()
    options, arguments = parser.parse_args()
    filters = getattr(options, "filters")
    monitor = getattr(options, "monitor")
    print(filters)
    print(monitor)
    main(filters, monitor)




