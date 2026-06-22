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

// GateCounterInstrumentationPass
// ───────────────────────────────
// Transformation pass that inserts runtime gate counters into a quantum circuit
// module.  For every unique (gate_name, num_wires) pair found in
// quantum.custom ops, a public memref.global of type memref<1xi64>
// initialised to 0 is added at module scope.  An increment sequence
// (memref.get_global / memref.load / arith.addi 1 / memref.store) is
// inserted immediately after each such op. quantum.measure ops are also
// counted under a Measure_1 counter.
//
// After the full compilation pipeline the memref globals become [1 x i64]
// LLVM globals accessible from Python via ctypes:
//
//     count = (ctypes.c_int64 * 1).in_dll(lib, "__gate_ctr_T_1")[0]
//
// A JSON manifest mapping gate_label → symbol_name is written to
// `manifestFile` so the Python side can discover which symbols exist.

#define DEBUG_TYPE "gate-counter-instrumentation"

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_GATECOUNTERINSTRUMENTATIONPASS
#define GEN_PASS_DEF_GATECOUNTERINSTRUMENTATIONPASS
#include "Catalyst/Transforms/Passes.h.inc"

// ── helpers ─────────────────────────────────────────────────────────────────

static std::string sanitizeIdent(llvm::StringRef s)
{
    std::string out;
    for (char c : s)
        out += (std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_';
    return out;
}

static std::string makeSymbol(const std::string &label)
{
    return "__gate_ctr_" + label;
}

// ── pass ────────────────────────────────────────────────────────────────────

struct GateCounterInstrumentationPass
    : public impl::GateCounterInstrumentationPassBase<GateCounterInstrumentationPass> {
    using GateCounterInstrumentationPassBase::GateCounterInstrumentationPassBase;

    void runOnOperation() final
    {
        ModuleOp mod = getOperation();
        MLIRContext *ctx = mod.getContext();
        OpBuilder builder(ctx);

        auto i64Ty = builder.getI64Type();
        auto memTy = MemRefType::get({1}, i64Ty);
        auto tenTy = RankedTensorType::get({1}, i64Ty);
        auto zeroAttr = DenseIntElementsAttr::get(tenTy, llvm::ArrayRef<int64_t>{0LL});

        // ── 1. Collect unique gate labels ─────────────────────────────
        // Use vector to preserve insertion order for the manifest.
        std::vector<std::pair<std::string, std::string>> gateList; // (label, symbol)
        std::set<std::string> seen;

        auto registerGate = [&](llvm::StringRef name, size_t wires) {
            auto label = sanitizeIdent(name) + "_" + std::to_string(wires);
            if (seen.insert(label).second)
                gateList.emplace_back(label, makeSymbol(label));
        };

        mod.walk([&](quantum::CustomOp op) {
            registerGate(op.getGateName(),
                         op.getInQubits().size() + op.getInCtrlQubits().size());
        });
        mod.walk([&](quantum::MeasureOp) { registerGate("Measure", 1); });

        if (gateList.empty())
            return;

        // ── 2. Create global counter variables at module scope ────────
        builder.setInsertionPointToStart(mod.getBody());
        for (auto &[label, sym] : gateList) {
            if (mod.lookupSymbol(sym))
                continue;
            memref::GlobalOp::create(
                builder, mod.getLoc(),
                builder.getStringAttr(sym),      // sym_name
                builder.getStringAttr("public"),  // sym_visibility
                TypeAttr::get(memTy),             // type
                zeroAttr,                         // initial_value
                UnitAttr(),                        // constant = absent → mutable
                IntegerAttr()                      // alignment = default
            );
            LLVM_DEBUG(llvm::dbgs() << "gate-counter: global " << sym << "\n");
        }

        // Fast lookup: label → symbol string
        std::map<std::string, std::string> labelToSym(gateList.begin(), gateList.end());

        // ── 3. Instrument quantum.custom ops ──────────────────────────
        SmallVector<quantum::CustomOp> customOps;
        mod.walk([&](quantum::CustomOp op) { customOps.push_back(op); });

        for (auto op : customOps) {
            auto label = sanitizeIdent(op.getGateName()) + "_" +
                         std::to_string(op.getInQubits().size() + op.getInCtrlQubits().size());
            builder.setInsertionPointAfter(op);
            insertIncrement(builder, op.getLoc(), labelToSym.at(label), memTy, i64Ty);
        }

        // ── 4. Instrument quantum.measure ops ─────────────────────────
        SmallVector<quantum::MeasureOp> measureOps;
        mod.walk([&](quantum::MeasureOp op) { measureOps.push_back(op); });

        for (auto op : measureOps) {
            builder.setInsertionPointAfter(op);
            insertIncrement(builder, op.getLoc(), labelToSym.at("Measure_1"), memTy, i64Ty);
        }

        // ── 5. Write JSON manifest ────────────────────────────────────
        if (!manifestFile.empty()) {
            std::ofstream out(manifestFile);
            if (out) {
                out << "{\n";
                for (size_t i = 0; i < gateList.size(); ++i) {
                    auto &[label, sym] = gateList[i];
                    out << "  \"" << label << "\": \"" << sym << "\"";
                    if (i + 1 < gateList.size())
                        out << ",";
                    out << "\n";
                }
                out << "}\n";
            }
            else {
                mod.emitWarning("gate-counter-instrumentation: cannot write manifest to ")
                    << manifestFile;
            }
        }

        markAllAnalysesPreserved();
    }

  private:
    // Insert load / addi 1 / store on the memref<1xi64> global named `sym`.
    void insertIncrement(OpBuilder &builder, Location loc, const std::string &sym,
                         MemRefType memTy, Type /*i64Ty*/)
    {
        Value ref = memref::GetGlobalOp::create(builder, loc, memTy, sym);
        SmallVector<Value> idx{arith::ConstantIndexOp::create(builder, loc, 0).getResult()};
        Value val = memref::LoadOp::create(builder, loc, ref, ValueRange(idx));
        Value one = arith::ConstantIntOp::create(builder, loc, 1, /*bitwidth=*/64);
        Value newVal = arith::AddIOp::create(builder, loc, val, one);
        memref::StoreOp::create(builder, loc, newVal, ref, ValueRange(idx));
    }
};

} // namespace catalyst
