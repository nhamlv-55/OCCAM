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

    def __init__(self, mode, workdir, atomizer = None, net = None, debug = False, p = -1, n = -1):
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
        self.spec_position  = p
        self.net = net
        self.debug = debug
        if self.debug: print("init QueryOracleServicer...")
        self.module_trace = {}
        self.mode = mode
        self.say_no = False
        self.say_yes = False
        self.atomizer = atomizer
        self.get_meta(workdir)
        self.rewriter = TU.IRTransformer(os.path.join(OCCAM_HOME, "razor/MLPolicy/inst2vec/published_results/data/vocabulary"), self.meta["struct_dict"])
    def get_meta(self, workdir):
        with open(os.path.join(workdir, "metadata.json"), "r") as metafile:
            self.meta = json.load(metafile)
    
    def handle_meta(self, meta, coloring = False):
        meta, worklist = meta.split("Worklist:\n")
        meta = meta.split("\n")
        callee_idx = meta.index("Callee:")
        caller_idx = meta.index("Caller:")
        if self.debug: print("caller_idx", caller_idx)
        if self.debug: print("callee_idx", callee_idx)
        callee = meta[callee_idx+1:caller_idx]
        caller = meta[caller_idx+1:-1]
        if self.debug: print("symbols2idx:", self.atomizer.idx2symbol)
        if self.debug: print("no of unique symboles:", len(self.atomizer.symbol2idx))
        meta_text = []
        module = meta[0]
        if module not in self.module_trace:
            self.module_trace[module] = {"spec_pos": len(self.module_trace)}
        worklist = worklist.split("\n")
        uses = []
        if coloring:
            for l in worklist:
                if "User:" not in l:
                    continue
                value, use = l.split("User:")
                value.strip()
                use.strip()
                uses.append(use)
                meta_text.append(colored(l, "yellow"))
            for l in meta:
                if l.strip()=="":
                    continue
                contain_use = False
                for u in uses:
                    if u in l:
                        if l.startswith("  br"):
                            meta_text.append(colored(l, "red"))
                        else:
                            meta_text.append(colored(l, "green"))
                        contain_use = True
                        break
                if not contain_use:
                    meta_text.append(l)
        else:
            meta_text = meta
        
        return module, meta_text, caller, callee

    def print_state(self, request):
        features = request.features
        features = [int(s) for s in features.split(',')]
        trace = np.array(request.trace)
        if len(features)!=len(self.names):
            print("Error in feature mapping")
            print("len features:", len(features))
            print("len names:", len(self.names))
            return
        trace = trace.reshape(-1, len(self.names))
        meta = request.meta
        caller = request.caller.splitlines()
        callee = request.callee.splitlines()

        stmt_index, rewritten_ir = self.rewriter.llvm_ir_to_input([caller, callee], ["caller", "callee"])
        rewritten_caller = rewritten_ir[0]
        rewritten_callee = rewritten_ir[1]
        
        # _, meta_text, caller, callee = self.handle_meta(meta)
        print("trace:")
        print(trace)
        #print("meta:")
        #for l in meta_text: print(l)
        print("caller:")
        for l in caller: print(l)
        print("rewritten caller:")
        for l in rewritten_caller: print(l)
        print("callee:")
        for l in callee: print(l)
        print("state:")
        for i in range(len(features)):
            print(self.names[i], ":", features[i])

    def Query(self, request, context):
        if self.debug: print(self.mode)
        if self.mode == Mode.INTERACTIVE:
            #immediately return results if flags are set
            if self.say_no:
                return Previrt_pb2.Prediction(q_no = -1, q_yes = -1, state_encoded = "empty", pred=False)
            if self.say_yes:
                return Previrt_pb2.Prediction(q_no = -1, q_yes = -1, state_encoded = "empty", pred=True)

            #normal pipeline
            self.print_state(request)
            pred = raw_input("Should I specialize? ([y]es/[n]o/[Y]es for this callsite and the rest/[N]o for this callsite and the rest)")
            if pred.strip()=="y":
                pred = True
            elif pred.strip()=="Y":
                pred = True
                self.say_yes = True
            elif pred.strip()=="N":
                pred = False
                self.say_no  = True
            else:
                pred = False
            return Previrt_pb2.Prediction(q_no = -1, q_yes = -1, state_encoded = "empty",pred=pred)
        elif self.mode == Mode.TRY_1_CS:
            meta = request.meta
            trace = np.array(request.trace)
            trace = trace.reshape(-1, len(self.names))
            module, meta_text, caller, callee = self.handle_meta(meta)
            state_encoded = self.atomizer.encode(meta.split("\n"))
            print(self.module_trace)
            if self.debug:
                for l in meta_text: print(l)
            if trace.shape[0]==self.spec_position:
                pred = True
            else:
                pred = False
            return Previrt_pb2.Prediction(q_no = -1, q_yes = -1, state_encoded = state_encoded,  pred=pred)
        elif self.mode == Mode.TRAINING:
            meta = request.meta
            _, _, caller, callee = self.handle_meta(meta)
            caller_encoded = self.atomizer.encode(caller)
            callee_encoded = self.atomizer.encode(callee)
            state_encoded = "caller:\n"+caller_encoded+"callee:\n"+callee_encoded+"----\n"
            if self.debug: self.print_state(request)
            features = [int(s) for s in request.features.split(',')]
            features = torch.FloatTensor([features])
            #print(features.shape)
            #print(self.net)
            logits = self.net.forward(features).view(-1).detach().numpy()
            if self.debug: print(logits)
            pred = np.random.choice([False, True], p=logits)
            if self.debug: print(pred)
            return Previrt_pb2.Prediction(q_no = logits[0], q_yes = logits[1], state_encoded = state_encoded, pred = pred)
        #context.set_trailing_metadata(('metadata_for_testint', b'I agree'),)
        elif self.mode == Mode.TRAINING_RNN:
            meta = request.meta
            caller = request.caller.splitlines()
            callee = request.callee.splitlines()
            stmt_index, rewritten_ir = self.rewriter.llvm_ir_to_input([caller, callee], ["caller", "callee"])
            rewritten_caller = rewritten_ir[0]
            rewritten_callee = rewritten_ir[1]

            encoded_caller = stmt_index[0]
            len_caller = len(encoded_caller)
            encoded_caller.extend([self.meta["padding_idx"]]*(self.meta["max_sequence_len"] - len(stmt_index[0])))

            encoded_callee = stmt_index[1]
            len_callee = len(encoded_callee)
            encoded_callee.extend([self.meta["padding_idx"]]*(self.meta["max_sequence_len"] - len(stmt_index[1])))

            encoded_args = [1]
            len_args = len(encoded_args)
            encoded_args.extend([0]*(self.meta["max_args_len"] - len_args))

            state_encoded = ""
            state_encoded +=str(len_caller)+" "
            state_encoded +=str(len_callee)+" "
            state_encoded +=str(len_args)+" "

            print("state_encoded with len caller and callee:", state_encoded)
            for e in encoded_caller:
                state_encoded+=str(e)+" "
            for e in encoded_callee:
                state_encoded+=str(e)+" "
            for e in encoded_args:
                state_encoded+=str(e)+" "
            state_encoded+="\n"
            #print(state_encoded)
            features = [len_caller, len_callee, len_args] + encoded_caller + encoded_callee + encoded_args     #
            features = torch.FloatTensor([features])
            print(features.shape)
            # #print(self.net)                                              #
            logits = self.net.forward(features).view(-1).detach().numpy() #
            print("logits:", logits)                                  #
            pred = np.random.choice([False, True], p=logits)              #
            # if self.debug: print(pred)                                    #
            return Previrt_pb2.Prediction(q_no = logits[0], q_yes = logits[1], state_encoded = state_encoded, pred = pred)
        else:
            return Previrt_pb2.Prediction(q_no = -1, q_yes = -1, state_encoded = "empty", pred=False)

def serve(mode, p, n, workdir):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(mode = mode, p = p, n = n, workdir = workdir), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

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

