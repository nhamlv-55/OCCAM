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

/**
 * Intra-module specialization.
 **/
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
//#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "SpecializationTable.h"
#include "Specializer.h"
/* here specialization policies */
#include "AggressiveSpecPolicy.h"
#include "RecursiveGuardSpecPolicy.h"
#include "MLPolicy.h"
/* call profiler */
#include "utils/Profiler.h"
#include "utils/QueryOracleClient.h"
using namespace llvm;
using namespace previrt;

static cl::opt<bool>
OptSpecialized("Ppeval-opt", cl::Hidden, cl::init(false),
               cl::desc("Optimize new specialized functions"));
static cl::opt<std::string> LogFilename("Ppeval-database", cl::desc("Specify output database filename"), cl::value_desc("database"));
static cl::opt<SpecializationPolicyType> SpecPolicy(
                                                    "Ppeval-policy", cl::desc("Intra-module specialization policy"),
                                                    cl::values(clEnumValN(NOSPECIALIZE, "nospecialize",
                                                                          "Skip intra-module specialization"),
                                                               clEnumValN(AGGRESSIVE, "aggressive",
                                                                          "Specialize always if some constant argument"),
                                                               clEnumValN(NONRECURSIVE_WITH_AGGRESSIVE, "nonrec-aggressive",
                                                                          "aggressive + non-recursive function"),
                                                               clEnumValN(ML, "machine-learning", "using machine learning policy")),
                                                    cl::init(ML));
static cl::opt<float> Epsilon("Ppeval-epsilon", cl::desc("Epsilon for MLPolicy"));

static cl::opt<bool> UseGRPC("Ppeval-grpc", cl::desc("Use GRPC mode"), cl::init(false));
namespace previrt {
  /* Intra-module specialization */
  class SpecializerPass : public llvm::ModulePass {
  private:
    bool optimize;
    bool trySpecializeFunction(Function *f, llvm::Module &M, SpecializationTable &table,
                               SpecializationPolicy *policy,
                               std::vector<Function *> &to_add);
    std::shared_ptr<Channel> connection = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
  public:
    static char ID;

    SpecializerPass(bool);
    virtual ~SpecializerPass();
    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnModule(llvm::Module &M);
    virtual llvm::StringRef getPassName() const {
      return "Intra-module specializer";
    }
  };
  /**
     Return true if any callsite in f is specialized using policy.
  **/
  bool SpecializerPass::trySpecializeFunction(Function *f, llvm::Module &M, SpecializationTable &table,
                                              SpecializationPolicy *policy,
                                              std::vector<Function *> &to_add) {
    QueryOracleClient client(connection);

    std::vector<Instruction *> worklist;
    //std::error_code EC;
    //raw_fd_ostream outputFile(llvm::StringRef("output_BB.bc"), EC, llvm::sys::fs::F_Append);

    for (BasicBlock &bb : *f) {
      for (Instruction &I : bb) {

        Instruction *ci = dyn_cast<CallInst>(&I);
        if (!ci)
          ci = dyn_cast<InvokeInst>(&I);
        if (!ci)
          continue;
        CallSite call(ci);

        Function *callee = call.getCalledFunction();
        if (!callee) {
          continue;
        }

        if (callee->isDeclaration() || callee->isVarArg()) {
          continue;
        }

        if (callee->hasFnAttribute(Attribute::NoInline) ||
            callee->hasFnAttribute(Attribute::OptimizeNone)) {
          continue;
        }

        worklist.push_back(ci);
      }
    }

    bool modified = false;

    while (!worklist.empty()) {
      Instruction *ci = worklist.back();
      worklist.pop_back();
      CallSite cs(ci);
      Function *callee = cs.getCalledFunction();
      Function *caller = cs.getCaller();
      assert(callee);
      if (!GlobalValue::isLocalLinkage(callee->getLinkage())) {
        // We only try to specialize a function if it's internal.
        continue;
      }
      // specScheme[i] = nullptr if the i-th parameter of the callsite
      //                         cannot be specialized.
      //                 c if the i-th parameter of the callsite is a
      //                   constant c
      std::vector<Value *> specScheme;
      bool specialize = false;
      if(SpecPolicy==ML){
        ProfilerPass &p = getAnalysis<ProfilerPass>();
        p.runOnModule(M);
        //const unsigned callee_no_of_uses = callee->getNumUses();
        const unsigned caller_no_of_uses = caller->getNumUses();
        //errs()<<"number of time callee is used:"<<callee_no_of_uses<<"\n";
        //errs()<<"number of time caller is used:"<<caller_no_of_uses<<"\n";
        //errs()<<"callee:"<<*callee<<"\n";
        //errs()<<"caller:"<<*caller<<"\n";
        //        errs()<<"Number of time callee is used:"<<no_of_uses<<std::endl;
        std::vector<float> module_features ;
        module_features.push_back((float)p.getNumFuncs());
        module_features.push_back((float)p.getTotalInst());
        module_features.push_back((float)p.getTotalBlocks());
        module_features.push_back((float)p.getTotalDirectCalls());
        //module_features.push_back((float)callee_no_of_uses);
        //module_features.push_back((float)worklist.size());
        module_features.push_back((float)caller_no_of_uses);

        specialize = policy->specializeOn(cs, specScheme, module_features, &client);
        //try dump policy
        //specialize = callee_no_of_uses>2 && specialize; 
      }else{
        specialize = policy->specializeOn(cs, specScheme);
      }
      if (!specialize) {
        continue;
      }

#if 1
      errs() << "Intra-specializing call to '" << callee->getName()
             << "' in function '" << ci->getParent()->getParent()->getName()
             << "' on arguments [";
      for (unsigned int i = 0, cnt = 0; i < callee->arg_size(); ++i) {
        if (specScheme[i] != NULL) {
          if (cnt++ != 0) {
            errs() << ",";
          }
          if (GlobalValue *gv =
              dyn_cast<GlobalValue>(cs.getInstruction()->getOperand(i))) {
            errs() << i << "=(@" << gv->getName() << ")";
          } else {
            errs() << i << "=(" << *cs.getInstruction()->getOperand(i) << ")";
          }
        }
      }
      errs() << "]\n";
      errs() << "before specialization:\n";
      //count number of instructions
      unsigned before_count = 0;
      for (const BasicBlock &BB : *callee){
        before_count += BB.size();
      }
      //errs() <<*callee;
      errs() << before_count;
      errs() << "\n";
#endif

      // --- build a specialized function if specScheme is more
      //     refined than all existing specialized versions.
      Function *specialized_callee = nullptr;
      std::vector<const SpecializationTable::Specialization *> versions;
      table.getSpecializations(callee, specScheme, versions);
      for (std::vector<const SpecializationTable::Specialization *>::iterator
             i = versions.begin(),
             e = versions.end();
           i != e; ++i) {
        if (SpecializationTable::Specialization::refines(specScheme,
                                                         (*i)->args)) {
          specialized_callee = (*i)->handle;
          break;
        }
      }

      if (!specialized_callee) {
        specialized_callee = specializeFunction(callee, specScheme);
        if (!specialized_callee) {
          errs()<<"failed to specialize\n";
          continue;
        }
        errs() <<"after specialization:\n";
        //count number of instructions after
        unsigned after_count = 0;
        for (const BasicBlock &BB : *specialized_callee){
          after_count += BB.size();
        }

        //errs() << *specialized_callee;
        errs() <<after_count;
        errs() <<"\n";
        table.addSpecialization(callee, specScheme, specialized_callee);
        to_add.push_back(specialized_callee);
      }

      // -- build the specialized callsite
      const unsigned int specialized_arg_count = specialized_callee->arg_size();
      std::vector<unsigned> argPerm;
      argPerm.reserve(specialized_arg_count);
      for (unsigned from = 0; from < callee->arg_size(); from++) {
        if (!specScheme[from]) {
          argPerm.push_back(from);
        }
      }
      assert(specialized_arg_count == argPerm.size());
      Instruction *newInst = specializeCallSite(ci, specialized_callee, argPerm);
      llvm::ReplaceInstWithInst(ci, newInst);
      modified = true;
    }

    

    return modified;
  }


  bool SpecializerPass::runOnModule(Module &M) {
    std::cerr<<"call runOnModule from IntraSpecializer.cpp::SpecializerPass"<<std::endl;
    // -- Create the specialization policy. Bail out if no policy.
    SpecializationPolicy *policy = nullptr;
    switch (SpecPolicy) {
    case NOSPECIALIZE:
      return false;
    case AGGRESSIVE:
      policy = new AggressiveSpecPolicy();
      break;
    case NONRECURSIVE_WITH_AGGRESSIVE: {
      SpecializationPolicy *subpolicy = new AggressiveSpecPolicy();
      CallGraph &cg = getAnalysis<CallGraphWrapperPass>().getCallGraph();
      policy = new RecursiveGuardSpecPolicy(subpolicy, cg);
      break;
    }
    case ML: {
      SpecializationPolicy *subpolicy = new AggressiveSpecPolicy();
      CallGraph &cg = getAnalysis<CallGraphWrapperPass>().getCallGraph();
      policy = new MLPolicy(subpolicy, cg, *this, LogFilename, Epsilon, UseGRPC);
      break;
    }
    }

    // -- Specialize functions defined in M
    std::vector<Function *> to_add;
    SpecializationTable table(&M);
    bool modified = false;
    for (auto &f : M) {
      if (f.isDeclaration())
        continue;
      modified |= trySpecializeFunction(&f, M, table, policy, to_add);
      //Run profiling after each function in M
      ProfilerPass &p = getAnalysis<ProfilerPass>();
      p.runOnModule(M);
      errs()<<"Profiling after each function in M\n";
      errs()<<p.getNumFuncs()<<" "<<p.getTotalInst()<<" "<<p.getTotalBlocks()<<" "<<p.getTotalDirectCalls();
      errs()<<"\n";
    }

    // -- Optimize new function and add it into the module
    llvm::legacy::FunctionPassManager *optimizer = nullptr;
    if (optimize) {
      optimizer = new llvm::legacy::FunctionPassManager(&M);
      // PassManagerBuilder builder;
      // builder.OptLevel = 3;
      // builder.populateFunctionPassManager(*optimizer);
    }

    while (!to_add.empty()) {
      Function *f = to_add.back();
      to_add.pop_back();
      if (f->getParent() == &M || f->isDeclaration()) {
        // The function was already in the module or
        // has already been added in this round of
        // specialization, no need to add it twice
        continue;
      }
      if (optimizer) {
        optimizer->run(*f);
      }
      M.getFunctionList().push_back(f);
    }

    if(SpecPolicy == ML){
      errs() << "Done 1 pass of MLPolicy. Early break. ";
    }
    if (modified) {
      errs() << "...progress...\n";
    } else {
      errs() << "...no progress...\n";
    }

    if (policy) {
      delete policy;
    }

    if (optimizer) {
      delete optimizer;
    }
    return modified;
  }

  void SpecializerPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<CallGraphWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ProfilerPass>();
    AU.setPreservesAll();
  }

  SpecializerPass::SpecializerPass(bool opt)
    : ModulePass(SpecializerPass::ID), optimize(opt) {
    errs() << "SpecializerPass(" << optimize << ")\n";
  }

  SpecializerPass::~SpecializerPass() {}

  class ParEvalOptPass : public SpecializerPass {
  public:
    ParEvalOptPass() : SpecializerPass(OptSpecialized.getValue()) {}
  };

  char SpecializerPass::ID;

} // end namespace previrt

static RegisterPass<previrt::ParEvalOptPass>
X("Ppeval", "Intra-module partial evaluation", false, false);
