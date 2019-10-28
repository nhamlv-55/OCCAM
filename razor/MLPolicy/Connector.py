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
from utils import *
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

    def __init__(self, mode, workdir, debug = True):
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
        self.get_meta(workdir)
        if self.debug: print("read Meta from metadata.json...")
        self._got_step = Event()
        if self.debug: print("create Event object...")
    def get_meta(self, workdir):
        with open(os.path.join(workdir, "metadata.json"), "r") as metafile:
            self.meta = json.load(metafile)

    def Step(self, request):
        self._predition = request.Predition
        self._got_step.set()
        print("got Step. Set event.")
        if os.path.exists(metric_file):
            reward = get_metrics(metric_file)
            done = True
        else:
            reward = 0
            done = False
        return Previrt_pb2.ORDI(obs = self._latest_obs, reward = reward, done = done, info = "EMPTY")

    def Query(self, request, context):
        if self.debug: print(self.mode)
        if self.mode == Mode.TRAINING:
            if self.debug: self.print_state(request)
            features = [int(s) for s in request.features.split(',')]
            print(features)
            self._latest_obs = features
            print("wait for Step")
            self._got_step.wait()
            self._got_step.clear()
            return self._predition
        else:
            return self._predition




def run_server(mode, workdir):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(mode = mode, workdir = workdir), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        pass

if __name__== "__main__":
    run_server(Mode.TRAINING, os.path.join(OCCAM_HOME, "examples/portfolio/tree"))
