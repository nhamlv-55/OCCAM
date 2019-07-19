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

from concurrent import futures
import time
import math
import logging

import grpc

import Previrt_pb2
import Previrt_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def get_action(state):
    """Returns Action at given location or None."""
    print("state[:10]", state.raw_code)
    return Previrt_pb2.Action(action = len(state.raw_code))

class QueryOracleServicer(Previrt_pb2_grpc.QueryOracleServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        pass

    def Query(self, request, context):
        action = get_action(request)
        if action is None:
            return Previrt_pb2.Action(action=-1)
        else:
            print(action)
            return Previrt_pb2.Action(action = 0)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    Previrt_pb2_grpc.add_QueryOracleServicer_to_server(
        QueryOracleServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()