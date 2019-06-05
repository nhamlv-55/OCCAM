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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "MLPolicy.h"

#include "iostream"
#include "llvm/ADT/SCCIterator.h"

using namespace llvm;
using namespace torch;
namespace previrt {

MLPolicy::MLPolicy(SpecializationPolicy *_delegate, CallGraph &_cg)
    : cg(_cg), delegate(_delegate) {

  assert(delegate);
  torch::Tensor tensor = torch::eye(3);
  std::cerr<< "Print a tensor"<< tensor <<std::endl;
  std::cerr << "Hello ML" << std::endl;
  // randomize weight
  //torch::nn::init::xavier_uniform_(this->net->fc1->weight, 1.0);
  //torch::nn::init::xavier_uniform_(this->net->fc2->weight, 1.0);
  std::cerr<<"w:::"<<this->net->fc1->weight<<std::endl;
  std::cerr<<"w:::"<<this->net->fc2->weight<<std::endl;

  markRecursiveFunctions();
}

MLPolicy::~MLPolicy() {
  if (delegate) {
    delete delegate;
  }
}

void MLPolicy::markRecursiveFunctions() {
  for (auto it = scc_begin(&cg); !it.isAtEnd(); ++it) {
    auto &scc = *it;
    bool recursive = false;

    if (scc.size() == 1 && it.hasLoop()) {
      // direct recursive
      recursive = true;
    } else if (scc.size() > 1) {
      // indirect recursive
      recursive = true;
    }

    if (recursive) {
      for (CallGraphNode *cgn : scc) {
        llvm::Function *fn = cgn->getFunction();
        if (!fn || fn->isDeclaration() || fn->empty()) {
          continue;
        }
        rec_functions.insert(fn);
      }
    }
  }
}

bool MLPolicy::isRecursive(llvm::Function *F) const {
  return rec_functions.count(F);
}

// Return true if F is not recursive
bool MLPolicy::allowSpecialization(llvm::Function *F) const {
  return (!isRecursive(F));
}

unsigned getInstructionCount(llvm::Function* f) {
    unsigned NumInstrs = 0;
    for (const BasicBlock &BB : *f)
      NumInstrs += BB.size();
    return NumInstrs;
 }

bool MLPolicy::specializeOn(CallSite CS, std::vector<Value *> &slice) const {
  llvm::Function *callee = CS.getCalledFunction();
  std::vector<float> features;
  features.push_back((float)CS.arg_size());
  features.push_back((float)getInstructionCount(callee));
  features.push_back(1.0);
  std::cerr<<"Feature vector: "<<features<<std::endl;

  torch::Tensor x = torch::tensor(at::ArrayRef<float>(features));
  //std::cerr<<"size x:"<<x<<std::endl;
  x = x.reshape({1, x.size(0)});
  //std::cerr<<"after reshaping"<<x<<std::endl;
  torch::Tensor prediction = this->net->forward(x);

  std::cerr<<"prediction: "<<prediction<<std::endl;
  if (callee && allowSpecialization(callee)) {
    return delegate->specializeOn(CS, slice);
  } else {
    return false;
  }
}

bool MLPolicy::specializeOn(llvm::Function *F, const PrevirtType *begin,
                            const PrevirtType *end,
                            SmallBitVector &slice) const {
  if (allowSpecialization(F)) {
    return delegate->specializeOn(F, begin, end, slice);
  } else {
    return false;
  }
}

bool MLPolicy::specializeOn(llvm::Function *F,
                            std::vector<PrevirtType>::const_iterator begin,
                            std::vector<PrevirtType>::const_iterator end,
                            SmallBitVector &slice) const {
  if (allowSpecialization(F)) {
    return delegate->specializeOn(F, begin, end, slice);
  } else {
    return false;
  }
}

} // end namespace previrt
