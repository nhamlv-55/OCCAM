#!/usr/bin/python
import Previrt_pb2 as pb
import sys
import os
cwd = os.getcwd()
if __name__=="__main__":
    filename = sys.argv[1]
    res = pb.ComponentInterface()
    res.ParseFromString(open(os.path.join(cwd, filename), "rb").read())
    print(res)
