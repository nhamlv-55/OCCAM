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

/*
 *  Initial implementation created on: Jul 11, 2011
 *  Author: malecha
 */

#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/CallSite.h"
#include "llvm/ADT/SmallBitVector.h"

#include "PrevirtTypes.h"

#include <vector>
#include "utils/QueryOracleClient.h"
namespace previrt
{
  
  /* Here specialization policies */
  enum SpecializationPolicyType {
    NOSPECIALIZE,
    AGGRESSIVE,
    NONRECURSIVE_WITH_AGGRESSIVE,
    ML
  };
  
  class SpecializationPolicy
  {
  protected:
    
    SpecializationPolicy(){}
    
    static bool isConstantSpecializable(llvm::Constant* cst) {
      if (!cst) return false;
      return PrevirtType::abstract(cst).isConcrete();
    }    
    
  public:
    
    virtual ~SpecializationPolicy(){}
    
    virtual bool specializeOn(llvm::CallSite, std::vector<llvm::Value*>&) const = 0;
    virtual bool specializeOn(llvm::CallSite,
                              std::vector<llvm::Value*>&,
                              QueryOracleClient* ,
                              const unsigned 
                              ) const = 0 ;


    virtual bool specializeOn(llvm::Function*,
			      const PrevirtType*, const PrevirtType*,
			      llvm::SmallBitVector&) const = 0;

    virtual bool specializeOn(llvm::Function*,
			      std::vector<PrevirtType>::const_iterator,
			      std::vector<PrevirtType>::const_iterator,
			      llvm::SmallBitVector&) const = 0;

  };
} // end namespace
