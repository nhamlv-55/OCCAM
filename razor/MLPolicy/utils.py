import os
import glob
import numpy as np
class Dataset:
    def __init__(self, folder, no_of_feats = 2):
        self.folder = folder
        self.no_of_feats = no_of_feats
        self.collect()
        
    
    def merge_csv(self, csv_files):
        dist = {}
        raw_data  = []
        input = []
        output = []
        total = 0
        for fname in csv_files:
            with open(fname, "r") as f:
                for l in f.readlines():
                    total+=1
                    tokens = l.strip().split(',')
                    tokens = [float(t) for t in tokens]
                    raw_data.append(tokens)
                    key = tuple(tokens[:self.no_of_feats])
                    label = tokens[-1]
                    if key in dist:
                        dist[key][int(label)]+=1
                    else:
                        dist[key]=[0,0]
                        dist[key][int(label)]+=1
        print(dist)
        for k in dist:
            label_0 = dist[key][0]
            label_1 = dist[key][1]
            label_0 = label_0/(label_0 + label_1)
            label_1 = 1 - label_0
            input.append(k)
            output.append((label_0, label_1))
        
        return raw_data, input, output

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
        print(runs)
        for r in runs:
            run_data = {}
            print(r)
            csv_files = glob.glob(r+"/*.csv")
            _, input, output = self.merge_csv(csv_files)
            result = self.get_stat(r)
            run_data["input"] = input
            run_data["output"] = output
            run_data["score"] = self.score(result)
            run_data["raw_result"] = result
            self.all_data.append(run_data)
        self.all_data.sort(key=lambda x: x["score"])


    def dump(self):
        for r in self.all_data:
            for i in range(len(r["input"])):
                print(r["input"][i], ":", r["output"][i])
            print(r["score"])
if __name__== "__main__":
    dataset = Dataset("/Users/e32851/workspace/OCCAM/examples/portfolio/tree/slash", no_of_feats = 8)
    dataset.dump()
