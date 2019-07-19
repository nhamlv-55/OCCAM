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

#include <random>
#include "MLPolicy.h"
#include "llvm/ADT/SCCIterator.h"
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include "llvm/Analysis/LoopInfo.h"
using namespace llvm;
using namespace torch;
namespace previrt {

  MLPolicy::MLPolicy(SpecializationPolicy *_delegate, CallGraph &_cg,
                     llvm::Pass &_pass, std::string _database, const float _epsilon)
    : cg(_cg), delegate(_delegate), pass(_pass), epsilon(_epsilon){
    database->assign(_database);
    assert(delegate);
    torch::Tensor tensor = torch::eye(3);
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

  // return true if a random number between 0 and 1 is less than prob
  bool MLPolicy::random_with_prob(const double prob) const {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(0, 1); // define the range

    //return true;
    double sample = distr(eng);
    std::cerr << "sample:" << sample << " threshold:" << prob<< std::endl;
    return sample <= prob;
  }

  // Return true if F is not recursive
  bool MLPolicy::allowSpecialization(llvm::Function *F) const {
    return (!isRecursive(F));
  }

  std::vector<float> normalizeFeatures(const std::vector<float> features){
    std::vector<float> normalized;
    return normalized;
  }


  std::vector<float> MLPolicy::getInstructionCount(llvm::Function *f) const {
    std::vector<float> counts;
    float total_int_count = 0;
    float total_bb_count = 0;
    float load_int_count = 0;
    float store_int_count = 0;
    float call_int_count = 0;
    float branch_int_count = 0;

    for (const BasicBlock &BB : *f){
      total_bb_count ++;
      total_int_count += BB.size();
      for (const Instruction &I: BB){
        if(llvm::isa<llvm::LoadInst>(I)) load_int_count++;
        if(llvm::isa<llvm::StoreInst>(I)) store_int_count++;
        if(llvm::isa<llvm::CallInst>(I)) call_int_count++;
        if(llvm::isa<llvm::BranchInst>(I)) branch_int_count++;
      }
    }

    counts.push_back(total_bb_count);
    counts.push_back(total_int_count);
    counts.push_back(load_int_count);
    counts.push_back(store_int_count);
    counts.push_back(call_int_count);
    counts.push_back(branch_int_count);

    counts.push_back((float)getLoopCount(f));

    return counts;
  }

  unsigned handleLoop(llvm::Loop *L, unsigned loopcounter) {
    loopcounter++;
    for (llvm::Loop *SL : L->getSubLoops()) {
      loopcounter += handleLoop(SL, loopcounter);
    }
    return loopcounter;
  }

  float MLPolicy::getLoopCount(llvm::Function *f) const {
    float loopcounter = 0;

    llvm::LoopInfo &li = pass.getAnalysis<LoopInfoWrapperPass>(*f).getLoopInfo();
    for (llvm::LoopInfo::iterator LIT = li.begin(), LEND = li.end(); LIT != LEND;
         ++LIT) {
      loopcounter = handleLoop(*LIT, loopcounter);
    }
    return loopcounter;
  }

  void MLPolicy::pushToTrace(const int v) const{
    std::vector<int> onehot = std::vector<int>(2, 0);
    onehot[v] = 1;
    trace->insert(trace->end(), onehot.begin(), onehot.end());
  //   for(std::vector<int>::iterator it = trace->begin(); it !=trace->end(); it++,i++ )    {
  //     // found nth element..print and break.
  //     if(*it == 0) {
  //       *it = v;
  //       break;
  //     }
  //   }
  }

  bool MLPolicy::specializeOn(llvm::CallSite, std::vector<llvm::Value*>&) const {return false;};
  bool MLPolicy::specializeOn(CallSite CS,
                              std::vector<Value *> &slice,
                              const std::vector<float> module_features) const {
    std::cerr<<"TOUCH A CALL SITE"<<std::endl;
    std::cerr<<"EPSILON:"<<epsilon<<std::endl;
    //const int type = 0; //Policy gradient
    const int type = 1; // DQN
    //const int type = 2; //AggressiveSpecPolicy
    llvm::Function *callee = CS.getCalledFunction();
    llvm::Function *caller = CS.getCaller();
    float q_Yes = -1;
    float q_No = -1;
    float no_of_arg = CS.arg_size();
    float no_of_const = 0;
    if (callee && allowSpecialization(callee)) {
      // directly borrow from AggressiveSpecPolicy
      bool specialize = false;
      std::vector<unsigned> argument_features;
      slice.reserve(CS.arg_size());
      for (unsigned i = 0, e = CS.arg_size(); i < e; ++i) {
        Constant *cst = dyn_cast<Constant>(CS.getArgument(i));
        // XXX: cst can be nullptr
        if (SpecializationPolicy::isConstantSpecializable(cst)) {
          slice.push_back(cst);
          argument_features.push_back(1);
          specialize = true;
          no_of_const++;
        } else { 
          slice.push_back(nullptr);
          argument_features.push_back(0);
        }
      }
      // return false immediately
      if(specialize==false){std::cerr<<"all arguemnts are not specializable"<<std::endl; return false;}
      // only invoke MLPolicy after this point
      std::vector<float> features;
      std::vector<float> callee_features = getInstructionCount(callee);
      std::vector<float> caller_features = getInstructionCount(caller);
      features.insert( features.end(), callee_features.begin(), callee_features.end() );
      features.insert( features.end(), caller_features.begin(), caller_features.end() );
      features.push_back(no_of_const);
      features.push_back(no_of_arg);
      features.insert( features.end(), module_features.begin(), module_features.end() );
      llvm::Module  *M = CS.getParent()->getModule();
      //      features.push_back((float)M->getInstructionCount ());
      //features.insert( features.end(), (*trace).begin(), (*trace).end());
      //std::vector<float> trace_mask = std::vector<float>(42-(*trace).size(), 0);
      //features.insert( features.end(), trace_mask.begin(), trace_mask.end());
      //features.insert( features.end(), argument_features.begin(), argument_features.end());
      std::cerr << "trace so far:"<<(*trace)<<std::endl;
      std::cerr << "Feature vector: " << features << std::endl;
      std::cerr << "Invoke MLpolicy" <<std::endl;
      std::cerr << "Module feature: " << module_features <<std::endl;
      //      return false;
      //return random_with_prob(0.5);
      bool final_decision;
      if(!random_with_prob(epsilon) || (type==1)){ //if random<epsilon -> random, if not, call the policy. for DQN, always use policy)
        torch::Tensor x = torch::tensor(at::ArrayRef<float>(std::vector<float>(features.begin(), features.end())));
        x = x.reshape({1, x.size(0)});
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x);
        std::cerr << x << std::endl;
        at::Tensor prediction = module->forward(inputs).toTensor();
        q_No  = prediction[0][0].item<float>();
        q_Yes = prediction[0][1].item<float>();
        switch(type){
        case 0: //Policy Gradient
          final_decision = random_with_prob(q_Yes);
          break;
        case 1: //DQN
          if(random_with_prob(epsilon))
            final_decision = random_with_prob(0.5);
          else
            final_decision = q_Yes > q_No;
          break;
        case 2: //AggressiveSpecPolicy
          final_decision = true;
          break;
        default:
          final_decision = random_with_prob(0.5);
        }
      }else{
        q_Yes = -1;
        q_No = -1;
        switch(type){
        case 2:
          final_decision = true;
          break;
        default:
          final_decision = random_with_prob(0.5); 
        }
      }
      //record data to file
      for (double f: features){ s->append(std::to_string(f).append(",")); }
      s->append(std::to_string(q_No));
      s->append(",");
      s->append(std::to_string(q_Yes));
      s->append(",");
      s->append(std::to_string((int)final_decision));
      s->append("\n");
      //note: currently +1 because we use a fixed size vector for trace with value 0 used as mask.
      pushToTrace((int)final_decision);
      return final_decision;
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
