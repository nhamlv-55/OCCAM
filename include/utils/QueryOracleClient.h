#pragma once
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <proto/Previrt.pb.h>
#include <proto/Previrt.grpc.pb.h>
#include "llvm/Support/raw_ostream.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

class QueryOracleClient {
public:
  QueryOracleClient(std::shared_ptr<Channel> channel)
    : stub_(previrt::proto::QueryOracle::NewStub(channel)) {
  }

  bool Query(previrt::proto::State s ) {
    llvm::errs()<<"call query with s="<<s.features()<<"\n";
    previrt::proto::Prediction p;

    // Connection timeout in seconds
    unsigned int client_connection_timeout = 100;

    ClientContext context;

    // Set timeout for API
    std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(client_connection_timeout);

    context.set_deadline(deadline);

    Status status = stub_->Query(&context, s, &p);
    if (!status.ok()) {
      llvm::errs() << "Query rpc failed.\n";
      return false;
    }
    return p.pred();
  }

  previrt::proto::State MakeState(const std::string& features, const std::string& meta, const std::vector<float>& trace) {
    previrt::proto::State s;
    s.set_features(features);
    s.set_meta(meta);
    //throw trace to proto message
    for(size_t i = 0; i < trace.size(); i++){
      s.add_trace((int)trace[i]);
    }
    return s;
  }
private:
  std::unique_ptr<previrt::proto::QueryOracle::Stub> stub_;


};


