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

#ifndef PURL_DIALECT
#define PURL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Purl dialect definition.
//===----------------------------------------------------------------------===//

def Purl_Dialect : Dialect {
    let name = "purl";
    let summary = "A small self-contained dialect for the Purl loop-cut op.";
    let description = [{
        Purl cuts carry-type dynamic quantum loops to bound the carried wire's
        coherent depth. `--purl` (analysis + rewrite) emits one abstract cut op,
        `purl.qcut`, at each periodic cut site with the chosen strategy baked in;
        `--purl-lower-qcut` mechanically expands it. The dialect is deliberately
        separate from the core `quantum` dialect -- it only reuses `!quantum.bit`
        and the quantum observable attribute.
    }];
    let cppNamespace = "::catalyst::purl";

    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Purl operation definition.
//===----------------------------------------------------------------------===//

class Purl_Op<string mnemonic, list<Trait> traits = []> :
        Op<Purl_Dialect, mnemonic, traits>;

#endif // PURL_DIALECT
