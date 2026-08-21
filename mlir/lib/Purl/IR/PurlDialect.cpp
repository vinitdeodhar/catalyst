// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Purl/IR/PurlDialect.h"

#include "llvm/ADT/TypeSwitch.h" // needed for generated attribute parser
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated attribute parser

#include "Purl/IR/PurlOps.h"

using namespace mlir;
using namespace catalyst::purl;

#include "Purl/IR/PurlOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Purl dialect.
//===----------------------------------------------------------------------===//

void PurlDialect::initialize()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "Purl/IR/PurlAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Purl/IR/PurlOps.cpp.inc"
        >();
}

#define GET_ATTRDEF_CLASSES
#include "Purl/IR/PurlAttributes.cpp.inc"
