import Previrt_pb2
import Previrt_pb2_grpc
import grpc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', help ='ip and port of the connector that we want to notify, eg: localhost:50051')

args = parser.parse_args()
connection = args.c
if __name__ == "__main__":
    with grpc.insecure_channel(connection) as channel:
        stub = Previrt_pb2_grpc.QueryOracleStub(channel)
        response = stub.Done(Previrt_pb2.Empty())
        print(response)
