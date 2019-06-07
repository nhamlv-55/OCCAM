import os
import glob
import numpy as np
class DataCollector:
    def __init__(self, folder):
        self.folder = folder

    def merge_csv(self, csv_files):
        data  = []
        for fname in csv_files:
            with open(fname, "r") as f:
                for l in f.readlines():
                    tokens = l.strip().split(',')
                    data.append([float(t) for t in tokens])
        return data

    def get_stat(self, run):
        result = []
        with open(run+"/0_results.txt", "r") as f:
            for l in f.readlines():
                if "[" in l:
                    continue
                result.append(int(l.strip().split()[0]))
        return result
            
    def score(self, result):
        return result[3]
    

    def collect(self):
        all_data = []
        runs = glob.glob(self.folder+"/run*")
        print(runs)
        for r in runs:
            print(r)
            csv_files = glob.glob(r+"/*.csv")
            data = self.merge_csv(csv_files)
            result = self.get_stat(r)
            all_data.append((data, result, self.score(result)))
        all_data.sort(key=lambda x: x[2])
        print(all_data)

if __name__== "__main__":
    dataCollector = DataCollector("/Users/e32851/workspace/OCCAM/examples/portfolio/tree/slash")
    dataCollector.collect()
