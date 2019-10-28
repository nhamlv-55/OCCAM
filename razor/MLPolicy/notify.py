import Previrt_pb2
import Previrt_pb2_grpc
import grpc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p', help ='port of the connector that we want to notify')

args = parser.parse_args()
port = args.p
if __name__ == "__main__":
    with grpc.insecure_channel('localhost:'+port) as channel:
        stub = Previrt_pb2_grpc.QueryOracleStub(channel)
        response = stub.Done(Previrt_pb2.Empty())
        print(response)
