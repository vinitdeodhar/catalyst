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

#include "Purl/IR/PurlOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Purl/IR/PurlDialect.h"

#include "Purl/IR/PurlEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Purl/IR/PurlOps.cpp.inc"

using namespace mlir;
using namespace catalyst::purl;

//===----------------------------------------------------------------------===//
// QCutOp
//===----------------------------------------------------------------------===//

LogicalResult QCutOp::verify()
{
    const bool isKnit = getStrategy() == Strategy::knit;
    const bool hasWeightIn = static_cast<bool>(getInWeight());
    const bool hasWeightOut = static_cast<bool>(getOutWeight());

    if (isKnit && !(hasWeightIn && hasWeightOut)) {
        return emitOpError("`knit` strategy requires an f64 weight operand and result");
    }
    if (!isKnit && (hasWeightIn || hasWeightOut)) {
        return emitOpError("`refresh` strategy must not carry a weight");
    }
    return success();
}
