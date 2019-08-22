#include "utils/UberIRDumper.h"

#include <set>

namespace previrt {
namespace utils {

  using namespace llvm;
  bool dump_IR_as_tokens(const Instruction& I, raw_string_ostream* s){
    *s << I.getOpcodeName()<< " ";
    for(unsigned i = 0; i < I.getNumOperands(); ++i){
        Value *opnd = I.getOperand(i);
        if (opnd->hasName()) {
          *s << " "<< opnd->getName().str();
          //          std::cout << " " << o << "," ;
        } else {
          *s << " ptr" << opnd;
        }
    }
    *s << "\n";
    return true;

  }
  
  bool dump_IR_as_tokens(const BasicBlock& BB, raw_string_ostream* s){
    *s << "  BasicBlock " << BB.getName().str() << "\n";
    for (const Instruction &I : BB) {
      dump_IR_as_tokens(I, s);
    }
    return true; 
  }

  bool dump_IR_as_tokens(const Function& F, raw_string_ostream* s){
    *s << "  Function " << F.getName().str() << "\n";
    for (const BasicBlock &BB : F) {
      dump_IR_as_tokens(BB, s);
    }
    return true;
  }

  bool dump_IR_as_tokens(const Module& M, raw_string_ostream* s){
    for (const Function &F: M) {
      dump_IR_as_tokens(F, s);
    }
    return true;
  }
}
}
