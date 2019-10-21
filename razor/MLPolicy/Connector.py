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
import torch
from utils import *
from net import * 
from termcolor import colored
from inst2vec import task_utils as TU 
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

    def __init__(self, mode, workdir, debug = False):
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
        self.rewriter = TU.IRTransformer(os.path.join(OCCAM_HOME, "razor/MLPolicy/inst2vec/published_results/data/vocabulary"), self.meta["struct_dict"])
        self._got_step = Event()
    def get_meta(self, workdir):
        with open(os.path.join(workdir, "metadata.json"), "r") as metafile:
            self.meta = json.load(metafile)

    def Step(self, request):
        self._predition = request.Predition
        self.got_step.set()
        if os.path.exists(metric_file):
            reward = get_metrics(metric_file)
            done = True
        else:
            reward = 0
            done = False
        return Previrt_pb2.Obs_rew_done_info(obs = self._latest_obs, reward = reward, done = done, info = "EMPTY")

    def Query(self, request, context):
        if self.debug: print(self.mode)
        if self.mode == Mode.TRAINING:
            if self.debug: self.print_state(request)
            features = [int(s) for s in request.features.split(',')]
            self._latest_obs = features
            self._got_step.wait()
            self._got_step.clear()
            return self._predition
        else:
            return self._predition




def _run_server(mode, p, n, workdir, net):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options = options)
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(mode = mode, p = p, n = n, workdir = workdir, net = net), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def serve_multiple(no_of_servers, mode, p, n, workdir, net):
        workers = []
        for _ in range(no_of_servers):
            # NOTE: It is imperative that the worker subprocesses be forked before
            # any gRPC servers start up. See
            # https://github.com/grpc/grpc/issues/16001 for more details.
            worker = multiprocessing.Process(
                target=_run_server, args = (mode, p, n, workdir, net))
            worker.start()
            workers.append(worker)
        return workers
#        for worker in workers:
#            worker.join()

def try_1_cs(p):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(mode = Mode.TRY_1_CS, p = p,  debug = False), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    _ = subprocess.check_output(("./build.sh -g -epsilon 0 -folder %s"%str(p)).split(), cwd = "/home/workspace/OCCAM/examples/portfolio/tree")
    server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default = 10, type=int, help='position to specialize')
    parser.add_argument('-n', default = 3, type=int, help ='module to specialize')
    parser.add_argument('-mode', default = 'interactive', type=str, help = 'grpc mode: training, interactive, try_1_cs')
    parser.add_argument('-workdir', default = '/home/workspace/OCCAM/examples/portfolio/tree')
    args = parser.parse_args()
    p = args.p
    n = args.n
    mode = args.mode
    workdir= args.workdir
    if mode=='training':
        mode = Mode.TRAINING
    elif mode=='interactive':
        mode = Mode.INTERACTIVE
    elif mode == 'try_1_cs':
        mode = Mode.TRY_1_CS
    elif mode == 'training_rnn':
        mode = Mode.TRAINING_RNN
    else:
        quit()
    #for i in range(21):
    #    print("spec_position:", i)
    #    try_1_cs(i)
    logging.basicConfig()
    serve(mode, p, n, workdir)

