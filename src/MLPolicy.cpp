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
#include "utils/UberIRDumper.h"
#include "utils/Profiler.h"
using namespace llvm;
namespace previrt {

  MLPolicy::MLPolicy(SpecializationPolicy *_delegate, CallGraph &_cg,
                     llvm::Pass &_pass, std::string _database, const float _epsilon, const bool _use_grpc)
    : cg(_cg), delegate(_delegate), pass(_pass), epsilon(_epsilon), use_grpc(_use_grpc){
    database->assign(_database);
    assert(delegate);
    errs() << "Hello ML" << "\n";
    // randomize weight
    // errs() << "w:::" << this->net->fc1->weight << "\n";
    // errs() << "w:::" << this->net->fc2->weight << "\n";
    markRecursiveFunctions();
  }

  MLPolicy::~MLPolicy() {
    if (delegate) {
      delete delegate;
    }
    pid_t pid = getpid();
    database->append(std::to_string(pid)).append("collected_data.csv");
    std::ofstream outFile(*database);
    errs() << "calling destructor with file " << *database << "\n";
    outFile << *s;
    outFile.flush();
    outFile.close();
    errs() << "file closed" << "\n";

    database->append(".state_encoded");
    std::ofstream outFile2(*database);
    errs() << "calling destructor with file " << *database << "\n";
    outFile2 << *state_encoded;
    outFile2.flush();
    outFile2.close();
    errs() << "file closed" << "\n";

  }

  void MLPolicy::markRecursiveFunctions() {
    errs() << "call markRecursiveFunctions\n";
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
    errs() << "sample:" << sample << " threshold:" << prob<< "\n";
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


  std::vector<unsigned> MLPolicy::getInstructionCount(llvm::Function *f) const {
    std::vector<unsigned> counts;
    unsigned total_int_count = 0;
    unsigned total_bb_count = 0;
    unsigned load_int_count = 0;
    unsigned store_int_count = 0;
    unsigned call_int_count = 0;
    unsigned branch_int_count = 0;

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

    counts.push_back(getLoopCount(f));

    return counts;
  }

  unsigned MLPolicy::getLoopCount(llvm::Function *f) const {
    unsigned TotalLoops = 0;
    LoopInfo& LI = pass.getAnalysis<LoopInfoWrapperPass>(*f).getLoopInfo();
    for (auto L: LI) {
      ++TotalLoops;
    }
    return TotalLoops;
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

  std::vector<unsigned> MLPolicy::getModuleFeatures(llvm::Function *caller) const {
    ProfilerPass &p = pass.getAnalysis<ProfilerPass>();
    p.runOnModule(*(caller->getParent()));
    //const unsigned callee_no_of_uses = callee->getNumUses();
    const unsigned caller_no_of_uses = caller->getNumUses();
    //errs()<<"number of time callee is used:"<<callee_no_of_uses<<"\n";
    //errs()<<"number of time caller is used:"<<caller_no_of_uses<<"\n";
    //errs()<<"callee:"<<*callee<<"\n";
    //errs()<<"caller:"<<*caller<<"\n";
    //        errs()<<"Number of time callee is used:"<<no_of_uses<<std::endl;
    std::vector<unsigned> module_features ;
    module_features.push_back(p.getNumFuncs());
    module_features.push_back(p.getTotalInst());
    module_features.push_back(p.getTotalBlocks());
    module_features.push_back(p.getTotalDirectCalls());
    //module_features.push_back((float)callee_no_of_uses);
    module_features.push_back(caller_no_of_uses);
    return module_features;
  }

  bool MLPolicy::specializeOn(llvm::CallSite, std::vector<llvm::Value*>&) const {return false;};

  bool countAllUsage(Value* arg, llvm::raw_string_ostream* rso, float& branch_cnt, float& affected_inst){
    std::vector<Value *> worklist;
    std::vector<Value *> visited;
    worklist.push_back(arg);
    while(!worklist.empty()){
      Value *e = worklist.back();
      affected_inst++;
      worklist.pop_back();
      visited.push_back(e);
      //errs()<<"processing ";
      //e->print(errs());
      //errs()<<"...\n";
      for(auto U : e->users()){  // U is of type User*
        //e->print(errs());
        //e->print(*rso);
        if (auto I = dyn_cast<Instruction>(U)){
          if(std::find(visited.begin(), visited.end(), U) == visited.end()) {
            //errs()<<"\tUser:";
            //*rso<<"\tUser:";
            //I->print(errs());
            //I->print(*rso);
            //errs()<<"\n";
            //*rso<<"\n";
            worklist.push_back(I);
            if(isa<llvm::BranchInst>(*I)){
              branch_cnt++;
              //errs()<<"is a cmp inst\n";
            }
          }
        }
      }

    }
    return 0;
  }


  bool MLPolicy::specializeOn(CallSite CS,
                              std::vector<Value *> &slice,
                              QueryOracleClient* q,
                              const unsigned worklist_size) const {
    errs()<<"TOUCH A CALL SITE"<<"\n";
    errs()<<"EPSILON:"<<epsilon<<"\n";
    const int type = 0; //Policy gradient
    //const int type = 1; // DQN
    //const int type = 2; //AggressiveSpecPolicy
    const bool state_rnn = epsilon < -100;

    llvm::Function *callee = CS.getCalledFunction();
    llvm::Function *caller = CS.getCaller();
    float q_Yes = -1;
    float q_No = -1;
    float no_of_arg = CS.arg_size();
    float no_of_const = 0;
    float branch_cnt = 0;
    if (callee && allowSpecialization(callee)) {
      // directly borrow from AggressiveSpecPolicy
      bool specialize = false;
      std::string args_state = "";
      slice.reserve(CS.arg_size()); 
      //std::string user_str;
      std::string wl_str;
      //std::string tokens_str;
      //llvm::raw_string_ostream tokens_rso(tokens_str);
      //llvm::raw_string_ostream rso(user_str);
      llvm::raw_string_ostream wl_rso(wl_str);

      std::string calling_ctx;
      llvm::raw_string_ostream ctx_rso(calling_ctx);
      const int window_size = 5;
      wl_rso<<"Worklist:\n";
      //tokens_rso<<"Tokens:\n";
      //      rso<<"Module:";
      //      rso<<CS.getParent()->getParent()->getParent()->getModuleIdentifier();
      //      //dump raw IR caller and callee to grpc:
      //      rso<<"\n";
      //      rso<<"Callee:";
      //      rso<<callee->getName();
      //      rso<<"\n; Body:\n";
      //      callee->print(rso);
      //
      //      rso<<"Caller:";
      //      rso<<caller->getName();
      //      rso<<"\n; Body:\n";
      //      caller->print(rso);

      std::vector<Value*> worklist;
      float affected_inst = 0;
      
      for (unsigned i = 0, e = CS.arg_size(); i < e; ++i) {
        Constant *cst = dyn_cast<Constant>(CS.getArgument(i));
        // XXX: cst can be nullptr
        if (SpecializationPolicy::isConstantSpecializable(cst)) {
          //tokens_rso << "Const " << cst << "\n";
          slice.push_back(cst);
          args_state+="1 ";
          // count how many branch insts are affected
          unsigned arg_index = 0;
          for(auto arg = callee->arg_begin(); arg != callee->arg_end(); ++arg, ++arg_index) {
            if(arg_index==i){
              countAllUsage(arg, &wl_rso, branch_cnt, affected_inst);
              break;
            }
          }
          specialize = true;
          no_of_const++;
        } else { 
          slice.push_back(nullptr);
          args_state+="0 ";
        }
      }
      wl_rso<<"End worklist:\n";
      // return false immediately
      if(specialize==false){errs()<<"all arguemnts are not specializable"<<"\n"; return false;}
      // only invoke MLPolicy after this point
      std::vector<unsigned> features;
      std::vector<unsigned> callee_features = getInstructionCount(callee);
      features.insert( features.end(), callee_features.begin(), callee_features.end() );
      std::vector<unsigned> caller_features = getInstructionCount(caller);
      features.insert( features.end(), caller_features.begin(), caller_features.end() );
      features.push_back(no_of_const);
      features.push_back(no_of_arg);
      std::vector<unsigned> module_features = getModuleFeatures(caller);
      features.insert( features.end(), module_features.begin(), module_features.end() );
      features.push_back(branch_cnt);
      features.push_back(affected_inst);
      features.push_back(worklist_size);
      trace->insert(trace->end(), features.begin(), features.end());
      std::string caller_ir;
      llvm::raw_string_ostream caller_rso(caller_ir);
        
      std::string callee_ir;
      llvm::raw_string_ostream callee_rso(callee_ir);

      if(state_rnn){
        //build a calling context of k instructions before 
        Instruction* current = CS.getInstruction();
        for(int i=0 ;  i < window_size; i++){
          if(current==nullptr){
            ctx_rso<<"NONE\n";
            errs()<<"NONE\n";
          }else{
            current->print(errs());
            errs()<<"\n";
            current->print(ctx_rso);
            ctx_rso<<"\n";
            current = current->getPrevNode();
          }
        }
        //build a calling context of k instructions after
        current = CS.getInstruction();
        for(int i=0 ;  i < window_size; i++){
          if(current==nullptr){
            ctx_rso<<"NONE\n";
            errs()<<"NONE\n";
          }else{
            current->print(errs());
            errs()<<"\n";
            current->print(ctx_rso);
            ctx_rso<<"\n";
            current = current->getNextNode();
          }
        }

        caller->print(caller_rso);

        callee->print(callee_rso);


       
      }
      //      features.push_back((float)M->getInstructionCount ());
      //features.insert( features.end(), (*trace).begin(), (*trace).end());
      //std::vector<float> trace_mask = std::vector<float>(42-(*trace).size(), 0);
      //features.insert( features.end(), trace_mask.begin(), trace_mask.end());
      //features.insert( features.end(), argument_features.begin(), argument_features.end());
      // std::cerr << "trace so far:"<<(*trace)<<"\n";
      // std::cerr << "Feature vector: " << features << "\n";
      // std::cerr << "Invoke MLpolicy" <<"\n";
      // std::cerr << "Module feature: " << module_features <<"\n";
      bool final_decision;
      if(!random_with_prob(epsilon) || (type==1)){ //if random<epsilon -> random, if not, call the policy. for DQN, always use policy)
        if(use_grpc){
          std::stringstream ss;
          for(size_t i = 0; i < features.size(); ++i)
            {
              if(i != 0)
                ss << ",";
              ss << features[i];
            }
          std::string state = ss.str();
          //dump caller and callee tokens to grpc
          /////////////////////////////////////////////////////
          // tokens_rso << "Callee:\n";                      //
          // utils::dump_IR_as_tokens(*callee, &tokens_rso); //
          // tokens_rso << "Caller:\n";                      //
          // utils::dump_IR_as_tokens(*caller, &tokens_rso); //
          // tokens_rso<<"End tokens:\n";                    //
          /////////////////////////////////////////////////////

          previrt::proto::Prediction prediction = q->Query(q->MakeState(state, "meta", caller_rso.str(), callee_rso.str(), ctx_rso.str(), args_state, *trace));
          final_decision = prediction.pred();
          q_Yes = prediction.q_yes();
          q_No  = prediction.q_no();
          state_encoded->append(prediction.state_encoded());
          errs()<<"final_decision:"<<final_decision<<"\n";
        }
      }else{//not using policy or using grpc
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
      for (unsigned f: features){ s->append(std::to_string(f).append(",")); }
      s->append(std::to_string(q_No));
      s->append(",");
      s->append(std::to_string(q_Yes));
      s->append(",");
      s->append(std::to_string((int)final_decision));
      s->append("\n");
      //note: currently +1 because we use a fixed size vector for trace with value 0 used as mask.
      return final_decision;
    } else {
      errs() << "not callee or not allowSpecialization" << "\n";
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
