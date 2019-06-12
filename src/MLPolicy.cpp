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
#include "llvm/ADT/SCCIterator.h"
#include <random>
#include <sys/types.h>
#include <unistd.h>

#include "llvm/Analysis/LoopInfo.h"
using namespace llvm;
using namespace torch;
namespace previrt {

MLPolicy::MLPolicy(SpecializationPolicy *_delegate, CallGraph &_cg,
                   llvm::Pass &_pass, std::string _database)
    : cg(_cg), delegate(_delegate), pass(_pass) {
  database->assign(_database);
  assert(delegate);
  torch::Tensor tensor = torch::eye(3);
  //  std::cerr << "database:" << _database << std::endl;
  std::cerr << "Print a tensor" << tensor << std::endl;
  std::cerr << "Hello ML" << std::endl;
  // randomize weight
  // torch::nn::init::xavier_uniform_(this->net->fc1->weight, 1.0);
  // torch::nn::init::xavier_uniform_(this->net->fc2->weight, 1.0);
  // std::cerr << "w:::" << this->net->fc1->weight << std::endl;
  // std::cerr << "w:::" << this->net->fc2->weight << std::endl;
  markRecursiveFunctions();
}

MLPolicy::~MLPolicy() {
  if (delegate) {
    delete delegate;
  }
  pid_t pid = getpid();
  database->append(std::to_string(pid)).append("collected_data.csv");
  std::ofstream outFile(*database);
  std::cerr << "calling destructor with file " << *database << std::endl;
  outFile << *s;
  outFile.flush();
  outFile.close();
  std::cerr << "file closed" << std::endl;
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

unsigned MLPolicy::getInstructionCount(llvm::Function *f) const {
  unsigned NumInstrs = 0;
  for (const BasicBlock &BB : *f)
    NumInstrs += BB.size();
  return NumInstrs;
}

unsigned handleLoop(llvm::Loop *L, unsigned loopcounter) {
  loopcounter++;
  for (llvm::Loop *SL : L->getSubLoops()) {
    loopcounter += handleLoop(SL, loopcounter);
  }
  return loopcounter;
}

unsigned MLPolicy::getLoopCount(llvm::Function *f) const {
  unsigned loopcounter = 0;

  llvm::LoopInfo &li = pass.getAnalysis<LoopInfoWrapperPass>(*f).getLoopInfo();
  for (llvm::LoopInfo::iterator LIT = li.begin(), LEND = li.end(); LIT != LEND;
       ++LIT) {
    loopcounter = handleLoop(*LIT, loopcounter);
  }
  return loopcounter;
}

bool MLPolicy::specializeOn(CallSite CS, std::vector<Value *> &slice) const {
  llvm::Function *callee = CS.getCalledFunction();

  if (callee && allowSpecialization(callee)) {
    // setting up random devices and engines
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    float sample = dist(e2);
    // database->assign(std::to_string(sample));
    std::cerr << "sample:" << sample << std::endl;

    float threshold = 0.5; // sampling a random number. If it is less than
                           // threshold, specialize
    std::vector<float> features;
    features.push_back((float)CS.arg_size());
    features.push_back((float)getInstructionCount(callee));
    features.push_back((float)getLoopCount(callee));
    std::cerr << "Feature vector: " << features << std::endl;

    if (delegate->specializeOn(CS, slice)) {
      if (sample >
          0) { // use the policy if sample > k . k =0 means always use policy
        torch::Tensor x = torch::tensor(at::ArrayRef<float>(features));
        // std::cerr<<"size x:"<<x<<std::endl;
        x = x.reshape({1, x.size(0)});
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x);
        std::cerr << x << std::endl;
        // std::cerr<<"after reshaping"<<x<<std::endl;
        std::cerr << "call prediction" << std::endl;
        assert(module != nullptr);
        std::cerr << "ok\n";
        at::Tensor prediction = module->forward(inputs).toTensor();
        // torch::Tensor prediction = this->net->forward(x);

        std::cerr << "prediction: " << prediction << std::endl;
        sample = dist(e2);
        std::cerr << "sample: " << sample << std::endl;
        threshold = prediction[0][0].item<float>();
        for (float f : features)
          s->append(std::to_string(f)).append(",");
        for (int i = 0; i < 2; i++)
          s->append(std::to_string(prediction[0][i].item<float>())).append(",");
      } else {
        for (float f : features)
          s->append(std::to_string(f)).append(",");
        for (int i = 0; i < 2; i++)
          s->append("-1").append(",");
      }

      sample = dist(e2);
      std::cerr << sample << " --- " << threshold << std::endl;

      s->append(std::to_string(sample < threshold)).append("\n");

      if (sample < threshold) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }

  } else {
    std::cerr << "not callee or not allowSpecialization" << std::endl;
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
