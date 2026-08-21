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

// --purl-lower-qcut: mechanically expand each purl.qcut into its concrete op
// sequence (spec 3.7). Purely code-generation -- no analysis, reads only the op.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"
#include "Purl/IR/PurlOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace purl {

#define GEN_PASS_DEF_LOWERQCUTPASS
#include "Purl/Transforms/Passes.h.inc"

namespace {

// single-qubit gate helper
static Value gate(OpBuilder &b, Location loc, StringRef name, Value q)
{
    auto op = CustomOp::create(b, loc, name, ValueRange{q}, ValueRange{}, ValueRange{},
                               ValueRange{});
    return op.getOutQubits()[0];
}

// guarded single-wire transform: `scf.if cond { then(q) } else { q }`
static Value guarded(OpBuilder &b, Location loc, Value cond, Value q,
                     llvm::function_ref<Value(OpBuilder &, Value)> thenFn)
{
    auto ifOp = scf::IfOp::create(
        b, loc, cond,
        [&](OpBuilder &tb, Location l) { scf::YieldOp::create(tb, l, thenFn(tb, q)); },
        [&](OpBuilder &eb, Location l) { scf::YieldOp::create(eb, l, q); });
    return ifOp.getResult(0);
}

// the RNG hook: a private extern that samples one of the 8 (O,t) quasi terms.
static func::FuncOp getOrCreateSampleFn(Operation *anchor)
{
    auto mod = anchor->getParentOfType<ModuleOp>();
    if (auto fn = mod.lookupSymbol<func::FuncOp>("purl_sample_term"))
        return fn;
    OpBuilder b(mod.getBodyRegion());
    auto i64 = b.getI64Type(), i1 = b.getI1Type();
    auto fnTy = b.getFunctionType({}, {i64, i1});
    auto fn = func::FuncOp::create(b, mod.getLoc(), "purl_sample_term", fnTy);
    fn.setPrivate();
    return fn;
}

// REFRESH (gamma=1): measure (end segment) + reset to |0> + replay the captured
// |psi0> prep region. Weight-free; the expval upstream is untouched.
static void lowerRefresh(QCutOp op)
{
    OpBuilder b(op);
    Location loc = op.getLoc();
    Value inQ = op.getInQubit();
    Type qT = inQ.getType(), i1 = b.getI1Type();

    auto m = MeasureOp::create(b, loc, i1, qT, inQ, IntegerAttr());
    Value q0 = guarded(b, loc, m.getMres(), m.getOutQubit(),
                       [&](OpBuilder &gb, Value in) { return gate(gb, loc, "PauliX", in); });

    // inline the prep region on the reset qubit: clone its body mapping the region
    // block argument to q0, then take the yielded prepared qubit.
    Block &blk = op.getPrep().front();
    IRMapping map;
    map.map(blk.getArgument(0), q0);
    for (Operation &o : blk.without_terminator())
        b.clone(o, map);
    auto y = cast<YieldOp>(blk.getTerminator());
    Value prepared = map.lookupOrDefault(y.getQubit());

    op.getOutQubit().replaceAllUsesWith(prepared);
    op.erase();
}

// KNIT (gamma=4): the quasi-probability cut protocol (spec 3.4a-f), threading the
// f64 weight in_weight -> out_weight. The prep region is ignored (the eigenstate
// prep is sampled here from the RNG hook).
static void lowerKnit(QCutOp op)
{
    OpBuilder b(op);
    Location loc = op.getLoc();
    Type i1 = b.getI1Type();
    Value bit = op.getInQubit();
    Value wacc = op.getInWeight();
    Type qT = bit.getType();
    func::FuncOp sampleFn = getOrCreateSampleFn(op);

    // (a) sample one of the 8 (O,t) terms
    auto call = func::CallOp::create(b, loc, sampleFn, ValueRange{});
    Value bIdx = call.getResult(0); // i64 Pauli axis O
    Value tBit = call.getResult(1); // i1  eigenstate index t

    Value c0 = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
    Value c1 = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(1));
    Value c2 = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(2));
    Value fone = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(1.0));
    Value fneg = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(-1.0));
    Value ffour = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(4.0));
    auto eq = [&](Value a, Value c) {
        return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, a, c).getResult();
    };
    Value isX = eq(bIdx, c1);
    Value isY = eq(bIdx, c2);

    // (b) basis change: X -> H; Y -> adjoint-S then H
    Value q = guarded(b, loc, isX, bit,
                      [&](OpBuilder &tb, Value in) { return gate(tb, loc, "Hadamard", in); });
    q = guarded(b, loc, isY, q, [&](OpBuilder &tb, Value in) {
        Value s = gate(tb, loc, "S", in);
        return gate(tb, loc, "Hadamard", s);
    });

    // (c) measure -> outcome s
    auto meas = MeasureOp::create(b, loc, i1, qT, q, IntegerAttr());
    Value s = meas.getMres();
    q = meas.getOutQubit();

    // (d) reset to |0>
    q = guarded(b, loc, s, q,
                [&](OpBuilder &tb, Value in) { return gate(tb, loc, "PauliX", in); });

    // (e) prepare eigenstate |O,t>
    q = guarded(b, loc, tBit, q,
                [&](OpBuilder &tb, Value in) { return gate(tb, loc, "PauliX", in); });
    Value bxy = arith::OrIOp::create(b, loc, isX, isY);
    q = guarded(b, loc, bxy, q,
                [&](OpBuilder &tb, Value in) { return gate(tb, loc, "Hadamard", in); });
    q = guarded(b, loc, isY, q,
                [&](OpBuilder &tb, Value in) { return gate(tb, loc, "S", in); });

    // (f) signed term weight 4 * sigma * s_eff, folded into wacc
    Value sv = arith::SelectOp::create(b, loc, s, fneg, fone);
    Value isI = eq(bIdx, c0);
    Value sEff = arith::SelectOp::create(b, loc, isI, fone, sv);
    Value bNot0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, bIdx, c0);
    Value tAndB = arith::AndIOp::create(b, loc, tBit, bNot0);
    Value sig = arith::SelectOp::create(b, loc, tAndB, fneg, fone);
    Value wterm =
        arith::MulFOp::create(b, loc, arith::MulFOp::create(b, loc, ffour, sig), sEff);
    Value wn = arith::MulFOp::create(b, loc, wacc, wterm);

    op.getOutQubit().replaceAllUsesWith(q);
    op.getOutWeight().replaceAllUsesWith(wn);
    op.erase();
}

struct LowerQCutPass : impl::LowerQCutPassBase<LowerQCutPass> {
    using LowerQCutPassBase::LowerQCutPassBase;

    void runOnOperation() final
    {
        // collect first: we erase each op as we expand it.
        SmallVector<QCutOp> ops;
        getOperation()->walk([&](QCutOp op) { ops.push_back(op); });
        for (QCutOp op : ops) {
            if (op.getStrategy() == Strategy::refresh)
                lowerRefresh(op);
            else
                lowerKnit(op);
        }
    }
};

} // namespace
} // namespace purl
} // namespace catalyst
