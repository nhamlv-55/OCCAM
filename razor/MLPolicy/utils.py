from __future__ import print_function
import os
import glob
import numpy as np

class Dataset:
    def __init__(self, folder, no_of_feats = 2):
        self.folder = folder
        self.no_of_feats = no_of_feats
        self.collect()
        
    
    def merge_csv(self, csv_files):
        episode_data = []
        raw_data  = []

        input = []
        output = []
        total = 0
        for fname in csv_files:
            run = []
            with open(fname, "r") as f:
                for l in f.readlines():
                    if l.startswith("TOUCH A CALL"):
                        continue
                    total+=1
                    tokens = l.strip().split(',')
                    tokens = [float(t) for t in tokens]
                    raw_data.append(tokens)
                    key = tuple(tokens[:self.no_of_feats])
                    label = tokens[-1]
                    run.append((key, label))
            episode_data.append(run)
        return raw_data, episode_data, total

    def get_stat(self, run):
        result = {}
        with open(run+"/0_results.txt", "r") as f:
            for l in f.readlines():
                if "[" in l:
                    continue
                tokens = l.strip().split()
                v = int(tokens[0])
                k = " ".join(tokens[1:])
                result[k] = v
        return result
            
    def score(self, result):
        return result["Number of instructions"]
    

    def collect(self):
        self.all_data = []
        runs = glob.glob(self.folder+"/run*")
        sorted(runs)
        for r in runs:
            run_data = {}
            print(r)
            csv_files = glob.glob(r+"/*.csv")
            _, episode_data, total = self.merge_csv(csv_files)
            result = self.get_stat(r)
            run_data["episode_data"] = episode_data
            run_data["score"] = self.score(result)
            run_data["raw_result"] = result
            run_data["total"] = total
            self.all_data.append(run_data)
        self.all_data.sort(key=lambda x: x["score"])


    def dump(self):
        for r in self.all_data:
            print("score:", r["score"])
            print("number of call sites:", r["total"])
            print("number of passes:", len(r["episode_data"]))
            for run in r["episode_data"]:
                for callsite in run:
                    print(callsite)
                print("------")
        print("best score", self.all_data[0]["score"])
        print("worst score", self.all_data[-1]["score"])
if __name__== "__main__":
    OCCAM_HOME = os.environ['OCCAM_HOME']
    datapath = os.path.join(OCCAM_HOME, "examples/portfolio/tree/slash") 
    dataset = Dataset(datapath, no_of_feats = 14)
    dataset.dump()
