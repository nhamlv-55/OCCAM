//
// OCCAM
//
// Copyright (c) 2011-2018, SRI International
//
//  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of SRI International nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "SpecializationPolicy.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
// #include "torch/torch.h"
// #include "torch/script.h"
#include <vector>
#include <cstdlib>
#include <random>
#include "utils/QueryOracleClient.h"

namespace previrt
{
  
  /* This class takes as argument another specialization policy p.
     Specialize a callsite if the callee function is not recursive AND
     p also decides to specialize. */
  class MLPolicy : public SpecializationPolicy {
    
    typedef llvm::SmallSet<llvm::Function*, 32> FunctionSet;

    llvm::CallGraph& cg;
    llvm::Pass& pass;
    SpecializationPolicy* const delegate;
    std::string* database = new std::string();
    const float epsilon;
    std::string* s = new std::string();
    std::string* state_encoded = new std::string();
    std::vector<float>* trace = new std::vector<float>();
    // std::shared_ptr<torch::jit::script::Module> module = torch::jit::load( std::string(std::getenv("OCCAM_HOME")).append("/model.pt"));
    const bool use_grpc;
    FunctionSet rec_functions;
    
    void markRecursiveFunctions();
    bool isRecursive(llvm::Function* f) const;    
    bool allowSpecialization(llvm::Function* f) const;
    std::vector<unsigned> getInstructionCount(llvm::Function* f) const;
    unsigned getLoopCount(llvm::Function* f) const;
    std::vector<unsigned> getModuleFeatures(llvm::Function* caller) const;
    bool random_with_prob(const double prob) const;
    void pushToTrace(const int v) const;
    // Generate the net
    //Net* net = new Net();



  public:
    
    MLPolicy(SpecializationPolicy* delegate, llvm::CallGraph& cg, llvm::Pass& pass, std::string database, const float epsilon, const bool use_grpc);

    virtual ~MLPolicy();
    
    virtual bool specializeOn(llvm::CallSite CS,
                              std::vector<llvm::Value*>& slice
                              ) const override;

    virtual bool specializeOn(llvm::CallSite CS,
                              std::vector<llvm::Value*>& slice,
                              QueryOracleClient* q,
                              const unsigned worklist_size
                              ) const ;

    virtual bool specializeOn(llvm::Function* F,
			      const PrevirtType* begin,
			      const PrevirtType* end,
			      llvm::SmallBitVector& slice) const override;

    virtual bool specializeOn(llvm::Function* F,
			      std::vector<PrevirtType>::const_iterator begin,
			      std::vector<PrevirtType>::const_iterator end,
			      llvm::SmallBitVector& slice) const override;

  };
} // end namespace previrt

