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

import grpc

import Previrt_pb2
import Previrt_pb2_grpc
import numpy as np
import random
import torch
from utils import *
from net import * 
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_INTERACTIVE = False
#IMPORTANT: use 2>grpc_log to debug. Search for `calling application`
class QueryOracleServicer(Previrt_pb2_grpc.QueryOracleServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self, net = None, debug = False):
        self.names = [
           # "block_count",
           # "inst_count",
           # "load_inst_count",
           # "store_inst_count",
           # "call_inst_count",
           # "branch_inst_count",
           # "loop_count",
           # "no_of_const",
           # "no_of_args",
           #"M_no_of_funcs",
           #"M_no_of_insts",
           #"M_no_of_blocks",
           #"M_no_of_direct_calls",
           #"callee_no_of_use",
           # "caller_no_of_use",
           #"current_worklist_size",
            "branch_cnt"
        ]
        self.net = net
        self.debug = debug
        if self.debug: print("init QueryOracleServicer...")
        #self.policy = policy_type(workdir, model_path, FeedForwardSingleInputSoftmax)
        #self.policy.load(model_path)

    def print_state(self, request):
        features = request.features
        trace = np.array(request.trace)
        trace = trace.reshape(-1, len(self.names))
        features = [int(s) for s in features.split(',')]
        meta = request.meta
        print("trace:")
        print(trace)
        print("meta:")
        print(meta)
        print("state:")
        for i in range(len(features)):
            print(self.names[i], ":", features[i])

    def Query(self, request, context):
        if _INTERACTIVE:
            self.print_state(request)
            pred = raw_input("Should I specialize?")
            if pred.strip()=="y":
                pred = True
            else:
                pred = False
        else:
            if self.debug: self.print_state(request)
            features = [int(s) for s in request.features.split(',')]
            features = torch.FloatTensor([features])
            #print(features.shape)
            #print(self.net)
            logits = self.net.forward(features).view(-1).detach().numpy()
            if self.debug: print(logits)
            pred = np.random.choice([False, True], p=logits)
            
            if self.debug: print(pred)
        #context.set_trailing_metadata(('metadata_for_testint', b'I agree'),)
        return Previrt_pb2.Prediction(pred=pred)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()

##############################
# server = ...               #
# server.start()             #
# policy.run_policy(100, 10) #
# server.stop()              #
##############################
