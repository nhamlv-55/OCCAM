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

#include "SpecializationPolicy.h"

#include "torch/torch.h"
#define NO_OF_FEATS 3

struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(NO_OF_FEATS, 3));
    fc2 = register_module("fc2", torch::nn::Linear(3, 2));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), NO_OF_FEATS})));
    // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    //x = torch::log_softmax(x, /*dim=*/1);
    x = torch::softmax(fc2->forward(x), 1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


namespace previrt
{
  
  /* This class takes as argument another specialization policy p.
     Specialize a callsite if the callee function is not recursive AND
     p also decides to specialize. */
  class MLPolicy : public SpecializationPolicy {
    
    typedef llvm::SmallSet<llvm::Function*, 32> FunctionSet;

    llvm::CallGraph& cg;
    SpecializationPolicy* const delegate;
    FunctionSet rec_functions;
    
    void markRecursiveFunctions();
    bool isRecursive(llvm::Function* f) const;    
    bool allowSpecialization(llvm::Function* f) const;

    // Generate the net
    Net* net = new Net();
  public:
    
    MLPolicy(SpecializationPolicy* delegate, llvm::CallGraph& cg);

    virtual ~MLPolicy();
    
    virtual bool specializeOn(llvm::CallSite CS,
			      std::vector<llvm::Value*>& slice) const override;

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
