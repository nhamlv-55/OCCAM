# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide server."""
from __future__ import print_function
from concurrent import futures
import time
import math
import logging
from enum import Enum
import grpc
import os
import Previrt_pb2
import Previrt_pb2_grpc
import numpy as np
import random
import json
#from utils import *
import argparse
import multiprocessing
from threading import Event
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_INTERACTIVE = False
OCCAM_HOME = os.environ["OCCAM_HOME"]

class Mode(Enum):
    INTERACTIVE = 0
    TRAINING = 1
    TRY_1_CS = 2
    TRAINING_RNN = 3
#IMPORTANT: use 2>grpc_log to debug. Search for `calling application`
class QueryOracleServicer(Previrt_pb2_grpc.QueryOracleServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self, mode, workdir, idx, metric, debug = False):
        '''
        workdir is the original binary directory (../tree/)
        idx is the id of the run (0, 1, ...)
        rundir will be workdir/slash/runidx (.../tree/slash/run1)
        '''
        self.names = [
            "block_count",
            "inst_count",
            "load_inst_count",
            "store_inst_count",
            "call_inst_count",
            "branch_inst_count",
            "loop_count",
            "caller_block_count",
            "caller_inst_count",
            "caller_load_inst_count",
            "caller_store_inst_count",
            "caller_call_inst_count",
            "caller_branch_inst_count",
            "caller_loop_count",
            "no_of_const",
            "no_of_args",
            "M_no_of_funcs",
            "M_no_of_insts",
            "M_no_of_blocks",
            "M_no_of_direct_calls",
           # "callee_no_of_use",
            "caller_no_of_use",
            "current_worklist_size",
            "branch_cnt",
            "affected_inst"
        ]
        self.debug = debug
        if self.debug: print("init QueryOracleServicer...")
        self.mode = mode
        if self.debug: print("mode:", self.mode)
        self.get_meta(workdir)
        self.metadata["metric"] = metric
        if self.debug: print("read Meta from metadata.json...")
        self._got_step = Event()
        self._got_new_obs = Event()
        if self.debug: print("create Event object...")
        self._latest_obs = [-1]
        self.idx = idx
        self.rundir = os.path.join(workdir, "slash", "run"+idx, )
        self.done = False
    def get_metrics(self):
        result = {}
        #get rop gadgets count
        with open(os.path.join(self.rundir,"gadget_count.json"), "r") as f:
            data = json.load(f)
            for k in data:
                result[k]= data[k]
 
        #get instruction counts
        with open(os.path.join(self.rundir,"0AfterSpecialization_results.txt"), "r") as f:
            for l in f.readlines():
                if "[" in l:
                    continue
                tokens = l.strip().split()
                v = int(tokens[0])
                k = " ".join(tokens[1:])
                result[k] = v

        return result[self.metadata["metric"]]

    def get_meta(self, workdir):
        with open(os.path.join(workdir, "metadata.json"), "r") as metafile:
            self.metadata = json.load(metafile)

    def _reset(self):
        self._got_new_obs.clear()
        self._got_step.clear()
        self.done = False

    def Step(self, request, context):
        if request.q_yes != -99:
            self._prediction = request
            self._got_step.set()
            if self.debug: print("got Step. Set event.")
        else:
            if self.debug: print("got _get_obs")

        #only wait if the episode is not over
        self._got_new_obs.wait()
        self._got_new_obs.clear()

        if self.done:
            reward = self.get_metrics()
            done = True
            print("Episode is done! Reward = %s"%str(reward))
            self._reset()
        else:
            reward = 0
            done = False
        return Previrt_pb2.ORDI(obs = self._latest_obs, reward = reward, done = done, info = "EMPTY")

    def Query(self, request, context):
        print("got obs")
        if self.debug: print(self.mode)
        if self.mode == Mode.TRAINING:
            # if self.debug: self.print_state(request)
            features = [int(s) for s in request.features.split(',')]
            if self.debug: print(features)
            self._latest_obs = features
            self._got_new_obs.set()
            if self.debug: print("wait for Step")
            self._got_step.wait()
            self._got_step.clear()
            return self._prediction
        else:
            return self._prediction

    def Done(self, request, context):
        print("Being notified that the episode is over")
        self._got_new_obs.set()
        self.done = True
        
        return Previrt_pb2.Empty()


def run_server(mode, workdir, idx, metric):
    print("starting server with\n mode = %s\n workdir = %s\n idx = %s\n metric = %s"%(mode, workdir, idx, metric))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(mode = mode, workdir = workdir, idx = idx, metric = metric), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        pass

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--workdir', default=os.path.join(OCCAM_HOME, "examples/portfolio/tree/"), help='s')
parser.add_argument('-m', '--metric', default = 'Total unique gadgets')
parser.add_argument('-i', '--idx')
args = parser.parse_args()
workdir = args.workdir
metric = args.metric
idx = args.idx
if __name__== "__main__":
    run_server(Mode.TRAINING, workdir = workdir, idx = idx, metric = metric)
