#pragma once
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/BasicBlock.h"
#include <llvm/Support/raw_ostream.h>

using namespace llvm; 

namespace previrt {
  namespace utils {

    // Force to inline a function only if it belongs to inlined_functions.
    bool dump_IR_as_tokens(const Module& M, raw_string_ostream* s);
    bool dump_IR_as_tokens(const Function& F, raw_string_ostream* s);
    bool dump_IR_as_tokens(const BasicBlock& B, raw_string_ostream* s);
    bool dump_IR_as_tokens(const Instruction& I, raw_string_ostream* s);
  }
}
