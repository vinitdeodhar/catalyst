// Copyright 2024 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Loop-knitting pass: coherent-depth bounding of carry-type dynamic while loops.
// See loop-knitting/implementation_spec.md (Part 3). This file implements the
// analyses (Part 3.1 classification, 3.2 body depth, 3.3 cut-period window) and
// the rewrite (Part 3.4).

#define DEBUG_TYPE "loop-knit"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;
using llvm::SmallVector;
using llvm::Twine;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_LOOPKNITTINGPASS
#define GEN_PASS_DEF_LOOPKNITTINGPASS
#include "Quantum/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Calibration weights (Part 3.2)
//===----------------------------------------------------------------------===//
struct Calib {
    double gate1q = 1.0, gate2q = 1.0, readout = 1.0, tau = 1.0;
    double T1 = INFINITY, T2 = INFINITY;
    // error probabilities for the profitability cost model (3.5)
    double p1 = 0.0, p2 = 0.0, p_ro = 0.0, p_meas = 0.0;
    double p_leak = 0.0, p_leak_ro = 0.0, p_prep = 0.0; // non-transportable
    bool unit = true; // layer counting

    static Calib load(StringRef spec, int qubit = 0, double pLeak = 0.0)
    {
        Calib c;
        if (spec == "unit" || spec.empty())
            return c;
        c.unit = false;
        auto buf = llvm::MemoryBuffer::getFile(spec);
        if (!buf)
            return Calib{}; // fall back to unit weights on a missing file
        auto parsed = llvm::json::parse((*buf)->getBuffer());
        if (!parsed) {
            llvm::consumeError(parsed.takeError());
            return Calib{};
        }
        auto *obj = parsed->getAsObject();
        if (!obj)
            return c;
        auto get = [&](llvm::json::Object *o, StringRef k, double dflt) -> double {
            if (o)
                if (auto v = o->getNumber(k))
                    return *v;
            return dflt;
        };
        if (auto *qubits = obj->getArray("qubits")) {
            // per-qubit + coupling-map hardware dataset (e.g. IBM Eagle r3):
            // pull the carried wire's physical qubit and the median 2q error.
            c.gate1q = get(obj, "gate_1q_time", 32e-9);
            c.gate2q = get(obj, "gate_2q_time", 560e-9);
            c.readout = get(obj, "readout_time", 1.2e-6);
            c.tau = get(obj, "tau", 1e-6);
            c.p_prep = get(obj, "p_prep", 0.0);
            unsigned qi = (qubit >= 0 && (unsigned)qubit < qubits->size()) ? qubit : 0;
            auto *q = (*qubits)[qi].getAsObject();
            c.T1 = get(q, "T1", 250e-6);
            c.T2 = get(q, "T2", 150e-6);
            c.p1 = get(q, "gate_1q_err", 0.0);
            c.p_meas = get(q, "gate_1q_err", 0.0);
            c.p_ro = get(q, "readout_err", 0.0);
            // median 2q error over the coupling map
            SmallVector<double> e2;
            if (auto *edges = obj->getArray("edges"))
                for (auto &e : *edges)
                    if (auto *eo = e.getAsObject())
                        if (auto v = eo->getNumber("gate_2q_err"))
                            e2.push_back(*v);
            if (!e2.empty()) {
                std::sort(e2.begin(), e2.end());
                c.p2 = e2[e2.size() / 2];
            }
            c.p_leak = pLeak;             // separate knob (not in the dataset)
            c.p_leak_ro = pLeak * 0.5;
        }
        else {
            // flat calibration (durations + rates)
            c.gate1q = get(obj, "gate_1q", 30e-9);
            c.gate2q = get(obj, "gate_2q", 60e-9);
            c.readout = get(obj, "readout", 700e-9);
            c.tau = get(obj, "tau", 500e-9);
            c.T1 = get(obj, "T1", 150e-6);
            c.T2 = get(obj, "T2", 200e-6);
            c.p1 = get(obj, "p1", 0.0);
            c.p2 = get(obj, "p2", 0.0);
            c.p_ro = get(obj, "p_ro", 0.0);
            c.p_meas = get(obj, "p_meas", 0.0);
            c.p_leak = pLeak > 0.0 ? pLeak : get(obj, "p_leak", 0.0);
            c.p_leak_ro = get(obj, "p_leak_ro", 0.0);
            c.p_prep = get(obj, "p_prep", 0.0);
        }
        return c;
    }

    // per-op weight in the depth analysis
    double gate(unsigned nQubits) const { return nQubits >= 2 ? gate2q : gate1q; }
};

//===----------------------------------------------------------------------===//
// Small SSA-walk helpers
//===----------------------------------------------------------------------===//

// The output qubit of a CustomOp corresponding to input qubit `in` (matched by
// position among in/out qubit ranges). Returns null if `in` is not an in-qubit.
static Value customContinuation(CustomOp custom, Value in)
{
    auto ins = custom.getInQubits();
    auto outs = custom.getOutQubits();
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i] == in && i < outs.size())
            return outs[i];
    auto cins = custom.getInCtrlQubits();
    auto couts = custom.getOutCtrlQubits();
    for (size_t i = 0; i < cins.size(); ++i)
        if (cins[i] == in && i < couts.size())
            return couts[i];
    return nullptr;
}

// Is `v` (a measured qubit's result) reset -- i.e. consumed, a few gates
// downstream, by a PauliX correction (the reset pattern of Part 3.1)?
static bool hasPauliXDownstream(Value v)
{
    SmallVector<Value> work{v};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
        Value x = work.pop_back_val();
        if (!seen.insert(x).second)
            continue;
        for (Operation *user : x.getUsers()) {
            if (auto cu = dyn_cast<CustomOp>(user)) {
                if (cu.getGateName() == "PauliX")
                    return true;
                if (Value c = customContinuation(cu, x))
                    work.push_back(c);
            }
        }
    }
    return false;
}

enum class Slot { Carry, Reset, Unknown };

// Classify one quantum wire from block-arg/extract to its exit (Part 3.1):
// CARRY if it reaches insert/yield un-measured; RESET if measured then
// PauliX-corrected; UNKNOWN if measured without a provable reset.
static Slot classifyQubitLine(Value q)
{
    Value cur = q;
    llvm::DenseSet<Value> seen;
    while (cur && seen.insert(cur).second) {
        Operation *consumer = nullptr;
        for (Operation *user : cur.getUsers()) {
            if (isa<MeasureOp>(user) || isa<CustomOp>(user) || isa<InsertOp>(user)) {
                consumer = user;
                break;
            }
        }
        if (!consumer)
            return Slot::Carry; // reached yield/exit un-measured
        if (auto m = dyn_cast<MeasureOp>(consumer))
            return hasPauliXDownstream(m.getOutQubit()) ? Slot::Reset
                                                        : Slot::Unknown;
        if (auto cu = dyn_cast<CustomOp>(consumer)) {
            cur = customContinuation(cu, cur);
            continue;
        }
        return Slot::Carry; // InsertOp: threaded back to the register un-measured
    }
    return Slot::Carry;
}

// Backward-trace a boolean condition to see if it derives from a quantum.measure
// (following tensor/arith/stablehlo glue and one before-arg -> body-yield hop).
static bool conditionFromMeasure(scf::WhileOp loop)
{
    Block &before = loop.getBefore().front();
    auto cond = cast<scf::ConditionOp>(before.getTerminator());
    Block &body = loop.getAfter().front();
    auto yield = cast<scf::YieldOp>(body.getTerminator());

    SmallVector<Value> work{cond.getCondition()};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
        Value v = work.pop_back_val();
        if (!seen.insert(v).second)
            continue;
        // block argument of the before-region -> corresponding body yield value
        if (auto ba = dyn_cast<BlockArgument>(v)) {
            if (ba.getOwner() == &before) {
                unsigned idx = ba.getArgNumber();
                if (idx < yield.getResults().size())
                    work.push_back(yield.getResults()[idx]);
            }
            continue;
        }
        Operation *def = v.getDefiningOp();
        if (!def)
            continue;
        if (isa<MeasureOp>(def))
            return true;
        for (Value operand : def->getOperands())
            work.push_back(operand);
    }
    return false;
}

//===----------------------------------------------------------------------===//
// Pauli-frame proof of a known carried state (enables the cheap gamma=1 cut).
//
// Single-qubit Clifford tracking of the carried wire over the loop body: if its
// net action is a Pauli, the carried state at any cut is a KNOWN function of the
// classical loop state (|psi0> for identity; P^counter|psi0> for a fixed Pauli
// P), so the cut can be a deterministic measure-and-re-prepare (no
// quasi-probability weights, gamma = 1). Non-Clifford or entangling operations
// on the carried wire -> Unprovable (fall back to the gamma = 4 cut).
//===----------------------------------------------------------------------===//

enum class Known { Identity, KnownPauli, Unprovable };

struct FrameResult {
    Known kind = Known::Unprovable;
    char pauli = 'I'; // 'X'/'Y'/'Z' when kind == KnownPauli
};

// ---- multi-qubit Pauli in the X^x Z^z (Heisenberg-Weyl) representation ----
// Each wire carries (x,z) bits; the operator is prod_v X_v^x Z_v^z, with a sign
// (-1)^(fixed) * prod (-1)^(m_bit) over the outcome-bit parities in `odd`.
struct MP {
    llvm::DenseMap<Value, std::pair<bool, bool>> b; // wire -> (x,z)
    bool neg = false;
    SmallVector<int, 6> odd; // outcome bits contributing (-1)^m
};

static void oddToggle(MP &m, int bit)
{
    for (unsigned i = 0; i < m.odd.size(); ++i)
        if (m.odd[i] == bit) {
            m.odd.erase(m.odd.begin() + i);
            return;
        }
    m.odd.push_back(bit);
}

// left-multiply-in-place: a := a * o   (sign from Z_a1 X_o2 reordering)
static void mpMul(MP &a, const MP &o)
{
    a.neg ^= o.neg;
    for (int bit : o.odd)
        oddToggle(a, bit);
    for (auto &kv : o.b) {
        auto &ab = a.b[kv.first];
        // sign from moving Z^{z1} past X^{x2}: (-1)^{z1 * x2}
        if (ab.second && kv.second.first)
            a.neg ^= true;
        ab.first ^= kv.second.first;
        ab.second ^= kv.second.second;
        if (!ab.first && !ab.second)
            a.b.erase(kv.first);
    }
}

// do a and o anticommute?  parity of sum (x1 z2 + z1 x2)
static bool mpAnti(const MP &a, const MP &o)
{
    int s = 0;
    for (auto &kv : a.b) {
        auto it = o.b.find(kv.first);
        if (it == o.b.end())
            continue;
        s ^= (kv.second.first & it->second.second) ^
             (kv.second.second & it->second.first);
    }
    return s & 1;
}

// conjugate the (x,z) at wire v by a single-qubit Clifford; false if non-Clifford
static bool conjXZ(MP &m, StringRef g, Value v)
{
    auto it = m.b.find(v);
    if (it == m.b.end())
        return true;
    bool x = it->second.first, z = it->second.second;
    if (g == "Hadamard") {
        if (x && z)
            m.neg ^= true;
        std::swap(x, z);
    }
    else if (g == "S") {
        if (x && z)
            m.neg ^= true;
        z ^= x;
    }
    else if (g == "S_dagger" || g == "Sdg" || g == "SInverse") {
        if (x && !z)
            m.neg ^= true;
        z ^= x;
    }
    else if (g == "PauliX") {
        if (z)
            m.neg ^= true;
    }
    else if (g == "PauliZ") {
        if (x)
            m.neg ^= true;
    }
    else if (g == "PauliY") {
        if (x ^ z)
            m.neg ^= true;
    }
    else {
        return false; // non-Clifford
    }
    it->second = {x, z};
    if (!x && !z)
        m.b.erase(v);
    return true;
}

static void renameKey(MP &m, Value oldv, Value newv)
{
    auto it = m.b.find(oldv);
    if (it == m.b.end())
        return;
    auto v = it->second;
    m.b.erase(it);
    m.b[newv] = v;
}

// Prove the carried wire's net body action is a Pauli, allowing entangling gates
// with fresh measured ancillas (RUS-synthesis case). Stabilizer / Pauli-frame
// tracking: carry the carried wire's logical operators X_d, Z_d and the ancilla
// stabilizers as multi-qubit Paulis; conjugate through Clifford + CNOT/CZ; at an
// ancilla Z-measurement reduce (bail if a logical operator is disturbed); apply
// conditional-Pauli scf.if corrections. Net Pauli on the carried wire alone ->
// Identity / KnownPauli.
static FrameResult proveKnownState(Value carriedArg, Block &body)
{
    Value d = carriedArg;
    Value dcur = d;                  // current SSA value of the carried wire
    MP lx, lz;                       // logical X_d, Z_d
    lx.b[d] = {true, false};
    lz.b[d] = {false, true};
    SmallVector<MP, 4> stab;         // ancilla stabilizers
    llvm::DenseMap<Value, int> mbit; // measurement result i1 -> outcome bit id
    int nbit = 0;

    auto conjAll = [&](StringRef g, Value v) -> bool {
        if (!conjXZ(lx, g, v) || !conjXZ(lz, g, v))
            return false;
        for (MP &s : stab)
            if (!conjXZ(s, g, v))
                return false;
        return true;
    };
    // CNOT/CZ symplectic update on all rows, then rename in->out
    auto conjTwo = [&](bool cz, Value c, Value t, Value cOut, Value tOut) {
        auto upd = [&](MP &m) {
            auto ic = m.b.find(c);
            auto itt = m.b.find(t);
            bool xc = ic != m.b.end() && ic->second.first;
            bool zc = ic != m.b.end() && ic->second.second;
            bool xt = itt != m.b.end() && itt->second.first;
            bool zt = itt != m.b.end() && itt->second.second;
            if (cz) {                // CZ: z_c ^= x_t ; z_t ^= x_c
                zc ^= xt;
                zt ^= xc;
            }
            else {                   // CNOT(c,t): x_t ^= x_c ; z_c ^= z_t
                xt ^= xc;
                zc ^= zt;
            }
            auto setw = [&](Value w, bool x, bool z) {
                if (!x && !z)
                    m.b.erase(w);
                else
                    m.b[w] = {x, z};
            };
            setw(c, xc, zc);
            setw(t, xt, zt);
        };
        upd(lx);
        upd(lz);
        for (MP &s : stab)
            upd(s);
        renameKey(lx, c, cOut);
        renameKey(lx, t, tOut);
        renameKey(lz, c, cOut);
        renameKey(lz, t, tOut);
        for (MP &s : stab) {
            renameKey(s, c, cOut);
            renameKey(s, t, tOut);
        }
    };

    // measure wire a in Z (outcome bit `bit`); reduce. false if a logical op is
    // disturbed by a non-deterministic measurement.
    auto measureZ = [&](Value a, int bit) -> bool {
        MP za;
        za.b[a] = {false, true}; // Z_a
        // find a stabilizer anticommuting with Z_a (=> random outcome / coin)
        int pivot = -1;
        for (unsigned i = 0; i < stab.size(); ++i)
            if (mpAnti(stab[i], za)) {
                pivot = (int)i;
                break;
            }
        if (pivot >= 0) {
            MP g = stab[pivot];
            for (unsigned i = 0; i < stab.size(); ++i)
                if ((int)i != pivot && mpAnti(stab[i], za))
                    mpMul(stab[i], g);
            if (mpAnti(lx, za))
                mpMul(lx, g);
            if (mpAnti(lz, za))
                mpMul(lz, g);
            stab[pivot] = za; // new stabilizer is Z_a with the (unknown) outcome
        }
        else {
            // deterministic outcome: a logical op must not anticommute
            if (mpAnti(lx, za) || mpAnti(lz, za))
                return false;
        }
        // the logical ops now commute with Z_a: any residual Z_a component is the
        // outcome factor (-1)^bit -> drop wire a, fold parity into the sign.
        auto fold = [&](MP &m) {
            auto it = m.b.find(a);
            if (it != m.b.end() && !it->second.first && it->second.second) {
                oddToggle(m, bit);
                m.b.erase(it);
            }
        };
        fold(lx);
        fold(lz);
        return true;
    };

    // conditional single Pauli P on the carried wire d, guarded by outcome `bit`
    auto condPauli = [&](StringRef p, int bit) {
        // conjugating a logical op by P flips its sign iff P anticommutes with the
        // op's component on d; conditioned on `bit` -> toggle bit in odd.
        MP pp;
        bool px = (p == "PauliX" || p == "PauliY");
        bool pz = (p == "PauliZ" || p == "PauliY");
        pp.b[dcur] = {px, pz};
        if (mpAnti(lx, pp))
            oddToggle(lx, bit);
        if (mpAnti(lz, pp))
            oddToggle(lz, bit);
    };

    // walk the body ops in order
    for (Operation &op : body.without_terminator()) {
        if (auto ex = dyn_cast<ExtractOp>(op)) {
            // fresh ancilla |0> -> stabilizer Z on that wire
            MP s;
            s.b[ex.getQubit()] = {false, true};
            stab.push_back(s);
        }
        else if (auto cu = dyn_cast<CustomOp>(op)) {
            auto ins = cu.getInQubits();
            auto outs = cu.getOutQubits();
            if (ins.size() == 1) {
                if (!conjAll(cu.getGateName(), ins[0]))
                    return {Known::Unprovable, 'I'};
                renameKey(lx, ins[0], outs[0]);
                renameKey(lz, ins[0], outs[0]);
                for (MP &s : stab)
                    renameKey(s, ins[0], outs[0]);
                if (ins[0] == dcur)
                    dcur = outs[0];
            }
            else if (ins.size() == 2 &&
                     (cu.getGateName() == "CNOT" || cu.getGateName() == "CZ")) {
                conjTwo(cu.getGateName() == "CZ", ins[0], ins[1], outs[0], outs[1]);
                if (ins[0] == dcur)
                    dcur = outs[0];
                if (ins[1] == dcur)
                    dcur = outs[1];
            }
            else {
                return {Known::Unprovable, 'I'}; // e.g. Toffoli: out of scope
            }
        }
        else if (auto me = dyn_cast<MeasureOp>(op)) {
            int bit = nbit++;
            mbit[me.getMres()] = bit;
            if (!measureZ(me.getInQubit(), bit))
                return {Known::Unprovable, 'I'};
            renameKey(lx, me.getInQubit(), me.getOutQubit());
            renameKey(lz, me.getInQubit(), me.getOutQubit());
            for (MP &s : stab)
                renameKey(s, me.getInQubit(), me.getOutQubit());
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            // recognise `scf.if %m -> bit { P(%in) } else { %in }`
            if (ifOp.getNumResults() != 1 || ifOp.getElseRegion().empty())
                return {Known::Unprovable, 'I'};
            auto elseY = cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
            auto thenY = cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
            Value in = elseY.getOperand(0);
            Value res = ifOp.getResult(0);
            Value then = thenY.getOperand(0);
            StringRef pname;
            if (auto pc = then.getDefiningOp<CustomOp>()) {
                if (pc.getInQubits().size() != 1 || pc.getInQubits()[0] != in)
                    return {Known::Unprovable, 'I'};
                pname = pc.getGateName();
            }
            else if (then == in) {
                pname = ""; // no-op guard
            }
            else {
                return {Known::Unprovable, 'I'};
            }
            auto bitIt = mbit.find(ifOp.getCondition());
            if (in == dcur) { // the carried wire flows through this scf.if
                if (!pname.empty()) {
                    if (bitIt == mbit.end())
                        return {Known::Unprovable, 'I'};
                    if (pname != "PauliX" && pname != "PauliY" && pname != "PauliZ")
                        return {Known::Unprovable, 'I'};
                    condPauli(pname, bitIt->second);
                }
                renameKey(lx, in, res);
                renameKey(lz, in, res);
                dcur = res;
            }
            // else: ancilla-side conditional (e.g. reset) -- does not affect d
        }
        // AllocOp / InsertOp / DeallocOp: no effect on the logical frame
    }

    // net must be a Pauli on d alone, with outcome-independent sign
    auto onlyD = [&](MP &m, bool wantX, bool wantZ) -> bool {
        if (!m.odd.empty())
            return false; // outcome-dependent -> not a fixed Pauli
        for (auto &kv : m.b)
            if (kv.first != dcur)
                return false;
        auto it = m.b.find(dcur);
        bool x = it != m.b.end() && it->second.first;
        bool z = it != m.b.end() && it->second.second;
        return x == wantX && z == wantZ;
    };
    if (!onlyD(lx, true, false) || !onlyD(lz, false, true))
        return {Known::Unprovable, 'I'};
    bool nx = lx.neg, nz = lz.neg; // sign of X_d, Z_d images
    if (!nx && !nz)
        return {Known::Identity, 'I'};
    if (!nx && nz)
        return {Known::KnownPauli, 'X'};
    if (nx && !nz)
        return {Known::KnownPauli, 'Z'};
    return {Known::KnownPauli, 'Y'};
}

// Straight-line single-qubit prep applied to the carried qubit before the loop
// (e.g. H,T,H,T,H for |psi0>). Returns gate names in application order; ok=false
// if the prep is not a clean single-wire chain.
struct PrepChain {
    SmallVector<std::string> gates;
    bool ok = false;
};
static PrepChain captureInputPrep(Value initVal)
{
    PrepChain pc;
    Value v = initVal;
    SmallVector<std::string> rev;
    while (auto cu = v.getDefiningOp<CustomOp>()) {
        if (cu.getInQubits().size() != 1 || !cu.getInCtrlQubits().empty())
            return pc;
        rev.push_back(cu.getGateName().str());
        v = cu.getInQubits()[0];
    }
    // v should now be a fresh qubit (an ExtractOp result); assume |0>
    if (!v.getDefiningOp<ExtractOp>())
        return pc;
    for (auto it = rev.rbegin(); it != rev.rend(); ++it)
        pc.gates.push_back(*it);
    pc.ok = true;
    return pc;
}

//===----------------------------------------------------------------------===//
// Analysis result
//===----------------------------------------------------------------------===//
enum class Klass { Carry, Restart, Unknown };

struct LoopInfo {
    Klass klass = Klass::Unknown;
    SmallVector<int> carrySlots; // register slot indices classified CARRY
    int carrySlot = -1;          // the single carry slot, if exactly one
    int carryArgIdx = -1;        // body block-arg index of a bare-qubit carry
                                 // (-1 for register-threaded carries)
};

// Classify each quantum slot of the carry (Part 3.1).
static LoopInfo classify(scf::WhileOp loop)
{
    LoopInfo info;
    Block &body = loop.getAfter().front();

    bool anyUnknown = false, anyReset = false, anySlot = false;

    // Collect carried quantum values: block args of QuregType or QubitType.
    for (BlockArgument arg : body.getArguments()) {
        if (isa<QuregType>(arg.getType())) {
            // per-slot: each ExtractOp on this reg with a static index
            llvm::DenseMap<int64_t, Slot> slotState;
            body.walk([&](ExtractOp ex) {
                // does ex.getQreg() chain back to `arg`? (reg threaded by insert)
                Value r = ex.getQreg();
                llvm::DenseSet<Value> seen;
                while (r && seen.insert(r).second) {
                    if (r == arg)
                        break;
                    if (auto ins = r.getDefiningOp<InsertOp>())
                        r = ins.getInQreg();
                    else
                        r = nullptr; // fresh alloc or opaque: not the carried reg
                }
                if (r != arg)
                    return;
                auto slot = ex.getIdxAttr();
                if (!slot)
                    return;
                Slot s = classifyQubitLine(ex.getQubit());
                // combine multiple extracts of one slot: CARRY dominates, then
                // UNKNOWN, then RESET
                auto it = slotState.find(*slot);
                if (it == slotState.end())
                    slotState[*slot] = s;
                else if (s == Slot::Carry || it->second == Slot::Carry)
                    it->second = Slot::Carry;
                else if (s == Slot::Unknown || it->second == Slot::Unknown)
                    it->second = Slot::Unknown;
            });
            for (auto &kv : slotState) {
                anySlot = true;
                if (kv.second == Slot::Carry)
                    info.carrySlots.push_back((int)kv.first);
                else if (kv.second == Slot::Unknown)
                    anyUnknown = true;
                else
                    anyReset = true;
            }
        }
        else if (isa<QubitType>(arg.getType())) {
            anySlot = true;
            Slot s = classifyQubitLine(arg);
            if (s == Slot::Carry) {
                info.carrySlots.push_back(0); // bare-bit carry: single slot 0
                info.carryArgIdx = arg.getArgNumber();
            }
            else if (s == Slot::Unknown)
                anyUnknown = true;
            else
                anyReset = true;
        }
    }

    if (!info.carrySlots.empty()) {
        info.klass = Klass::Carry;
        if (info.carrySlots.size() == 1)
            info.carrySlot = info.carrySlots.front();
    }
    else if (anyUnknown || !anySlot) {
        info.klass = Klass::Unknown;
    }
    else if (anyReset) {
        info.klass = Klass::Restart;
    }
    return info;
}

//===----------------------------------------------------------------------===//
// Body depth B (Part 3.2)
//===----------------------------------------------------------------------===//
// Per-body-execution gate counts, for the profitability cost model (3.5).
struct BodyStats {
    int n1q = 0, n2q = 0, nro = 0;
};

static double bodyDepth(scf::WhileOp loop, const Calib &c, bool condMeas,
                        BodyStats *stats = nullptr)
{
    Block &body = loop.getAfter().front();
    llvm::DenseMap<Value, double> depth; // qubit/reg value -> depth
    auto d = [&](Value v) -> double {
        auto it = depth.find(v);
        return it == depth.end() ? 0.0 : it->second;
    };

    std::function<void(Block &)> visit = [&](Block &blk) {
        for (Operation &op : blk) {
            if (auto custom = dyn_cast<CustomOp>(op)) {
                double base = 0.0;
                for (Value q : custom.getInQubits())
                    base = std::max(base, d(q));
                for (Value q : custom.getInCtrlQubits())
                    base = std::max(base, d(q));
                unsigned n = custom.getInQubits().size() + custom.getInCtrlQubits().size();
                if (stats)
                    (n >= 2 ? stats->n2q : stats->n1q)++;
                double nd = base + c.gate(n);
                for (Value q : custom.getOutQubits())
                    depth[q] = nd;
                for (Value q : custom.getOutCtrlQubits())
                    depth[q] = nd;
            }
            else if (auto meas = dyn_cast<MeasureOp>(op)) {
                if (stats)
                    stats->nro++;
                double nd = d(meas.getInQubit()) + c.readout;
                depth[meas.getOutQubit()] = nd;
            }
            else if (auto ex = dyn_cast<ExtractOp>(op)) {
                depth[ex.getQubit()] = d(ex.getQreg());
            }
            else if (auto ins = dyn_cast<InsertOp>(op)) {
                depth[ins.getOutQreg()] =
                    std::max(d(ins.getInQreg()), d(ins.getQubit()));
            }
            else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                // recurse both regions; results get the per-branch max
                SmallVector<Region *, 2> regions{&ifOp.getThenRegion()};
                if (!ifOp.getElseRegion().empty())
                    regions.push_back(&ifOp.getElseRegion());
                for (Region *r : regions)
                    if (!r->empty())
                        visit(r->front());
                double bmax = 0.0;
                for (Region *r : regions) {
                    if (r->empty())
                        continue;
                    auto y = cast<scf::YieldOp>(r->front().getTerminator());
                    for (Value v : y.getResults())
                        bmax = std::max(bmax, d(v));
                }
                double add = condMeas ? c.tau : 0.0;
                for (Value res : ifOp.getResults())
                    depth[res] = bmax + add;
            }
        }
    };
    visit(body);

    double B = 0.0;
    auto yield = cast<scf::YieldOp>(body.getTerminator());
    for (Value v : yield.getResults())
        if (isa<QuregType>(v.getType()) || isa<QubitType>(v.getType()))
            B = std::max(B, d(v));
    return B;
}

//===----------------------------------------------------------------------===//
// Cut-period window (Part 3.3)
//===----------------------------------------------------------------------===//
struct Window {
    int cMin, cMax;
    bool empty() const { return cMin > cMax; }
};
static Window cutWindow(double p, double B, const Calib &c, double f)
{
    double gamma2 = 16.0;
    int cMin = (int)std::ceil(std::log(gamma2) / std::log(1.0 / (1.0 - p)));
    int cMax;
    if (std::isinf(c.T2))
        cMax = cMin + 2; // unit/layer mode: no coherence ceiling
    else
        cMax = (int)std::floor(f * c.T2 / (B + c.tau));
    return {cMin, cMax};
}

//===----------------------------------------------------------------------===//
// Time-based fidelity model (real hardware data). Predict the carried qubit's
// delivered fidelity after being held for D loop iterations.
//===----------------------------------------------------------------------===//

// count 1q/2q gates ON the carried wire per body execution
static void countCarriedGates(Value carriedArg, int &n1q, int &n2q)
{
    n1q = n2q = 0;
    Value cur = carriedArg;
    llvm::DenseSet<Value> seen;
    while (cur && seen.insert(cur).second) {
        Operation *consumer = nullptr;
        for (Operation *u : cur.getUsers())
            if (isa<MeasureOp>(u) || isa<CustomOp>(u) || isa<InsertOp>(u)) {
                consumer = u;
                break;
            }
        if (!consumer || isa<InsertOp>(consumer) || isa<MeasureOp>(consumer))
            break;
        auto cu = cast<CustomOp>(consumer);
        unsigned nq = cu.getInQubits().size() + cu.getInCtrlQubits().size();
        (nq >= 2 ? n2q : n1q)++;
        cur = customContinuation(cu, cur);
    }
}

// F(D) = e^{-t_idle/T1} e^{-t_idle/T2} (1-e1q)^{D n1q} (1-e2q)^{D n2q}
//        (1-leak)^{D n2q},  t_idle = D (B + tau)
static double predictFidelity(double D, const Calib &c, double Bsec, int n1q,
                              int n2q, double pLeak)
{
    double tidle = D * (Bsec + c.tau);
    double F = 1.0;
    if (!std::isinf(c.T1))
        F *= std::exp(-tidle / c.T1);
    if (!std::isinf(c.T2))
        F *= std::exp(-tidle / c.T2);
    F *= std::pow(1.0 - c.p1, D * n1q);
    F *= std::pow(1.0 - c.p2, D * n2q);
    F *= std::pow(1.0 - pLeak, D * n2q);
    return F;
}

// mean delivered fidelity over the geometric trip distribution. Unbounded: the
// carried qubit ages k iterations. Bounded/refresh at C: the delivered segment
// age is ((k-1) mod C) + 1 (reset every C).
static double meanFidelity(const Calib &c, double Bsec, int n1q, int n2q,
                           double pLeak, double p, int C /* 0 = unbounded */)
{
    double s = 0.0, w;
    for (int k = 1; k <= 200000; ++k) {
        w = p * std::pow(1.0 - p, k - 1);
        int age = C > 0 ? ((k - 1) % C) + 1 : k;
        s += w * predictFidelity((double)age, c, Bsec, n1q, n2q, pLeak);
        if (w < 1e-14 && k > 8)
            break;
    }
    return s;
}

//===----------------------------------------------------------------------===//
// Profitability cost model + strategy selection (Part 3.5). No DISCARD arm.
//===----------------------------------------------------------------------===//
enum class Strategy { None, Refresh, Knit };

struct Predicted {
    Strategy strat = Strategy::None;
    int C = 0;
    double none = 0, refresh = 0, knit = 0; // predicted expval errors
};

// per-iteration error rates split into transportable / non-transportable
struct EpsRates {
    double t = 0, nt = 0, cut = 0;
    double all() const { return t + nt; }
};
static EpsRates epsFromCalib(const Calib &c, double Bsec, const BodyStats &bs)
{
    EpsRates e;
    double idle = 0.0;
    if (!std::isinf(c.T1) && !std::isinf(c.T2))
        idle = (Bsec + c.tau) * (1.0 / c.T1 + 1.0 / c.T2);
    // transportable: idle decoherence + gate depolarizing + pre-measure depol
    e.t = idle + bs.n1q * c.p1 + bs.n2q * c.p2 + bs.nro * c.p_meas;
    // non-transportable: leakage per 2q gate and per readout
    e.nt = bs.n2q * c.p_leak + bs.nro * c.p_leak_ro;
    // cut overhead: a readout + a state preparation
    e.cut = c.p_ro + c.p_prep;
    return e;
}

// expected age of the delivered state under cutting every C iterations (3.5)
static double sbar(double p, int C)
{
    double s = 0.0, w;
    for (int j = 1; j <= 200000; ++j) {
        w = p * std::pow(1.0 - p, j - 1);
        s += w * (double)(((j - 1) % C) + 1);
        if (w < 1e-14 && j > 8)
            break;
    }
    return s;
}

static double rmse(double bias, double stat) { return std::hypot(bias, stat); }

// Select the profit-maximising strategy (3.5). tier1 = REFRESH is applicable.
static Predicted selectStrategy(double p, const EpsRates &e, const Window &win,
                                bool tier1, int shots, double sigma0,
                                double margin, int forceC)
{
    Predicted r;
    double Ek = 1.0 / p;
    double statBase = sigma0 / std::sqrt((double)std::max(shots, 1));

    // NONE
    r.none = rmse(e.all() * Ek, statBase);

    auto ecuts = [&](int C) {
        double q = std::pow(1.0 - p, C);
        return q / (1.0 - q);
    };

    // REFRESH (gamma = 1): C in [1, cMax], no variance floor
    double bestRef = INFINITY;
    int bestRefC = 0;
    if (tier1) {
        int lo = forceC > 0 ? forceC : 1;
        int hi = forceC > 0 ? forceC : std::max(1, win.cMax);
        for (int C = lo; C <= hi; ++C) {
            double bias = e.all() * sbar(p, C) + ecuts(C) * e.cut;
            double err = rmse(bias, statBase);
            if (err < bestRef) {
                bestRef = err;
                bestRefC = C;
            }
        }
    }
    r.refresh = bestRef;

    // KNIT (gamma = 4): C in [cMin, cMax], require 16 q < 1
    double bestKnit = INFINITY;
    int bestKnitC = 0;
    {
        int lo = forceC > 0 ? forceC : win.cMin;
        int hi = forceC > 0 ? forceC : win.cMax;
        for (int C = lo; C <= hi; ++C) {
            double q = std::pow(1.0 - p, C);
            if (16.0 * q >= 1.0)
                continue; // divergent variance
            double V = (1.0 - q) / (1.0 - 16.0 * q);
            double bias = e.t * Ek + e.nt * sbar(p, C) + ecuts(C) * e.cut;
            double stat = sigma0 * std::sqrt(V / (double)std::max(shots, 1));
            double err = rmse(bias, stat);
            if (err < bestKnit) {
                bestKnit = err;
                bestKnitC = C;
            }
        }
    }
    r.knit = bestKnit;

    // decision: arg-min applicable strategy, fire iff it beats NONE by margin
    double bestErr = INFINITY;
    if (bestRef < bestErr) {
        bestErr = bestRef;
        r.strat = Strategy::Refresh;
        r.C = bestRefC;
    }
    if (bestKnit < bestErr) {
        bestErr = bestKnit;
        r.strat = Strategy::Knit;
        r.C = bestKnitC;
    }
    if (!(bestErr * margin < r.none)) {
        r.strat = Strategy::None;
        r.C = 0;
    }
    return r;
}

//===----------------------------------------------------------------------===//
// Rewrite (Part 3.4) -- bare-qubit expval carry loop (the writeup's canonical
// input; its output is rus_marked_suspend.mlir).
//===----------------------------------------------------------------------===//

// single-qubit gate helper
static Value gate(OpBuilder &b, Location loc, StringRef name, Value q)
{
    auto op = CustomOp::create(b, loc, name, ValueRange{q}, ValueRange{},
                                 ValueRange{}, ValueRange{});
    return op.getOutQubits()[0];
}

// A guarded single-wire transformation: `scf.if cond { then(q) } else { q }`.
static Value guarded(OpBuilder &b, Location loc, Value cond, Value q,
                     llvm::function_ref<Value(OpBuilder &, Value)> thenFn)
{
    auto ifOp = scf::IfOp::create(b, 
        loc, cond,
        [&](OpBuilder &tb, Location l) {
            scf::YieldOp::create(tb, l, thenFn(tb, q));
        },
        [&](OpBuilder &eb, Location l) { scf::YieldOp::create(eb, l, q); });
    return ifOp.getResult(0);
}

// The expanded cut-and-reprepare protocol (Part 3.4a-f) on a bare qubit.
// Returns the fresh qubit and the accumulated weight.
static std::pair<Value, Value> buildCut(OpBuilder &b, Location loc, Value bit,
                                        Value wacc, func::FuncOp sampleFn)
{
    Type i1 = b.getI1Type();
    auto qT = bit.getType();

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
        return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, a, c)
            .getResult();
    };
    Value isX = eq(bIdx, c1);
    Value isY = eq(bIdx, c2);

    // (b) basis change: X -> H; Y -> adjoint-S then H
    Value q = guarded(b, loc, isX, bit,
                      [&](OpBuilder &tb, Value in) { return gate(tb, loc, "Hadamard", in); });
    q = guarded(b, loc, isY, q, [&](OpBuilder &tb, Value in) {
        Value s = gate(tb, loc, "S", in); // adjoint via S;S;S == S^-1 (schematic)
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
    Value bNot0 =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, bIdx, c0);
    Value tAndB = arith::AndIOp::create(b, loc, tBit, bNot0);
    Value sig = arith::SelectOp::create(b, loc, tAndB, fneg, fone);
    Value wterm = arith::MulFOp::create(b, 
        loc, arith::MulFOp::create(b, loc, ffour, sig), sEff);
    Value wn = arith::MulFOp::create(b, loc, wacc, wterm);
    return {q, wn};
}

// The post-loop expval chain on the carried qubit, if present (Part 3.4
// applicability precheck). axis: 0=Z(or I), 1=X, 2=Y.
struct ObsChain {
    ExpvalOp expval;
    NamedObsOp obs;
    int axis = 0;
    bool ok = false;
};
static ObsChain findObsChain(scf::WhileOp loop, int carryResultIdx)
{
    ObsChain oc;
    Value carried = loop.getResult(carryResultIdx);
    for (Operation *user : carried.getUsers()) {
        if (auto nobs = dyn_cast<NamedObsOp>(user)) {
            for (Operation *u2 : nobs.getObs().getUsers()) {
                if (auto ev = dyn_cast<ExpvalOp>(u2)) {
                    oc.obs = nobs;
                    oc.expval = ev;
                    switch (nobs.getType()) {
                    case NamedObservable::PauliX:
                        oc.axis = 1;
                        break;
                    case NamedObservable::PauliY:
                        oc.axis = 2;
                        break;
                    default:
                        oc.axis = 0;
                    }
                    oc.ok = true;
                    return oc;
                }
            }
        }
    }
    return oc;
}

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//
struct LoopKnittingPass : impl::LoopKnittingPassBase<LoopKnittingPass> {
    using LoopKnittingPassBase::LoopKnittingPassBase;

    void runOnOperation() final
    {
        // collect first: the rewrite erases/recreates loops, so we must not be
        // walking when that happens.
        SmallVector<scf::WhileOp> loops;
        getOperation()->walk([&](scf::WhileOp loop) {
            // only process outermost dynamic loops; a nested loop is reported by
            // its enclosing loop's nested-loop guard.
            if (!loop->getParentOfType<scf::WhileOp>())
                loops.push_back(loop);
        });
        for (scf::WhileOp loop : loops)
            processLoop(loop);
    }

    void processLoop(scf::WhileOp loop)
    {
        MLIRContext *ctx = &getContext();
        OpBuilder b(ctx);
        Calib c = Calib::load(calib, carryQubit, pLeak);
        int n1qCarry = 0, n2qCarry = 0; // gates on the carried wire per body

        // nested dynamic loop guard (Part 3.2)
        bool nested = false;
        loop.getAfter().walk([&](scf::WhileOp inner) {
            if (inner != loop)
                nested = true;
        });
        if (nested) {
            loop.emitError("nested dynamic loop: bound inner loop first");
            signalPassFailure();
            return;
        }

        LoopInfo info = classify(loop);
        StringRef klassStr = info.klass == Klass::Carry     ? "carry"
                             : info.klass == Klass::Restart ? "restart"
                                                            : "unknown";
        loop->setAttr("knit.class", b.getStringAttr(klassStr));

        if (info.klass == Klass::Unknown) {
            loop.emitRemark("loop-knit: unknown class; pass does not fire");
            return;
        }
        if (info.klass == Klass::Restart)
            return; // nothing to do (already coherent-depth bounded)

        // carry-type
        if (info.carrySlots.size() >= 2) {
            loop.emitError("multi-wire cut unsupported");
            signalPassFailure();
            return;
        }
        loop->setAttr("knit.carry_slot", b.getI64IntegerAttr(info.carrySlot));

        bool condMeas = conditionFromMeasure(loop);
        BodyStats bs;
        double B = bodyDepth(loop, c, condMeas, &bs);
        if (c.unit)
            loop->setAttr("knit.body_layers", b.getF64FloatAttr(B));
        else
            loop->setAttr("knit.body_seconds", b.getF64FloatAttr(B));

        // Tier-1 proof: can we prove the carried state is known (REFRESH, gamma=1)?
        FrameResult fr{Known::Unprovable, 'I'};
        PrepChain prep;
        bool tier1 = false;
        if (info.carryArgIdx >= 0) {
            Value carriedArg =
                loop.getAfter().front().getArgument(info.carryArgIdx);
            fr = proveKnownState(carriedArg, loop.getAfter().front());
            prep = captureInputPrep(loop.getInits()[info.carryArgIdx]);
            countCarriedGates(carriedArg, n1qCarry, n2qCarry);
            tier1 = (fr.kind != Known::Unprovable) && prep.ok;
            StringRef ks = fr.kind == Known::Identity     ? "identity"
                           : fr.kind == Known::KnownPauli ? "pauli"
                                                          : "none";
            loop->setAttr("knit.known_state", b.getStringAttr(ks));
        }

        Window wKnit = cutWindow(pSuccess, B, c, budgetFraction);

        // Strategy + cut-period selection.
        Strategy strat;
        int C = 0;
        if (shots > 0) {
            // Part 3.5: profitability model chooses NONE / REFRESH / KNIT and C.
            EpsRates e = epsFromCalib(c, B, bs);
            Predicted pred =
                selectStrategy(pSuccess, e, wKnit, tier1, shots, sigma0, margin,
                               cutPeriod > 0 ? cutPeriod : 0);
            strat = pred.strat;
            C = pred.C;
            auto f = [&](double x) {
                return b.getF64FloatAttr(std::isinf(x) ? -1.0 : x);
            };
            loop->setAttr("knit.predicted",
                          b.getDictionaryAttr({
                              b.getNamedAttr("none", f(pred.none)),
                              b.getNamedAttr("refresh", f(pred.refresh)),
                              b.getNamedAttr("knit", f(pred.knit)),
                          }));
        }
        else {
            // Legacy (no shot budget): fire the best available mechanism, no
            // profitability veto. REFRESH if tier-1, else KNIT.
            strat = tier1 ? Strategy::Refresh : Strategy::Knit;
            int cMin = strat == Strategy::Refresh ? 1 : wKnit.cMin;
            int cMax = wKnit.cMax;
            if (cMin > cMax) {
                loop.emitError("empty cut-period window: C_min=" + Twine(cMin) +
                               " > C_max=" + Twine(cMax));
                signalPassFailure();
                return;
            }
            int Cdef = strat == Strategy::Refresh ? std::min(3, cMax) : cMin;
            C = cutPeriod > 0 ? cutPeriod : Cdef;
            if (C < cMin || C > cMax) {
                loop.emitError("requested C=" + Twine(C) + " outside window [" +
                               Twine(cMin) + "," + Twine(cMax) + "]");
                signalPassFailure();
                return;
            }
        }

        StringRef stratStr = strat == Strategy::None      ? "none"
                             : strat == Strategy::Refresh ? "refresh"
                                                          : "knit";
        loop->setAttr("knit.strategy", b.getStringAttr(stratStr));
        loop->setAttr("knit.cut",
                      b.getStringAttr(strat == Strategy::Refresh ? "deterministic"
                                      : strat == Strategy::Knit  ? "quasiprobability"
                                                                 : "none"));
        int wLo = strat == Strategy::Refresh ? 1 : wKnit.cMin;
        if (strat != Strategy::None) {
            loop->setAttr("knit.C", b.getI64IntegerAttr(C));
            loop->setAttr("knit.window",
                          b.getDenseI64ArrayAttr({wLo, wKnit.cMax}));
        }

        // Time-based fidelity prediction from real hardware data (calib in
        // seconds). Predict the carried qubit's delivered fidelity for the
        // UNBOUNDED depth (mean trip count E[k]=1/p) and the BOUNDED depth (the
        // cut period C the pass would enforce).
        if (!c.unit) {
            // bounded uses the strategy's C if firing, else the tightest bound
            // the best applicable mechanism could use.
            int Cb = strat != Strategy::None ? C
                     : tier1                 ? 1
                                             : wKnit.cMin;
            double Fu = meanFidelity(c, B, n1qCarry, n2qCarry, pLeak, pSuccess, 0);
            double Fb =
                meanFidelity(c, B, n1qCarry, n2qCarry, pLeak, pSuccess, Cb);
            loop->setAttr("knit.predicted_fidelity",
                          b.getDictionaryAttr({
                              b.getNamedAttr("unbounded", b.getF64FloatAttr(Fu)),
                              b.getNamedAttr("bounded", b.getF64FloatAttr(Fb)),
                          }));
            if (depth > 0)
                loop->setAttr(
                    "knit.fidelity_at_depth",
                    b.getF64FloatAttr(predictFidelity((double)depth, c, B,
                                                      n1qCarry, n2qCarry, pLeak)));
        }

        if (analyzeOnly)
            return;

        // Idempotence guard
        if (loop->hasAttr("knit.applied"))
            return;

        if (strat == Strategy::None) {
            loop.emitRemark("loop-knit: not profitable; loop left unchanged");
            return;
        }
        if (info.carryArgIdx < 0) {
            loop.emitRemark("loop-knit: register-threaded carry; analyses only");
            return;
        }
        Window wSel{wLo, wKnit.cMax};
        bool ok = strat == Strategy::Refresh
                      ? doRewriteDeterministic(loop, info.carryArgIdx, C, wSel, fr, prep)
                      : doRewrite(loop, info.carryArgIdx, C, wSel);
        if (!ok)
            loop.emitRemark("loop-knit: output is not an expval of the carried "
                            "qubit; rewrite skipped");
    }

    func::FuncOp getOrCreateSampleFn(Operation *anchor)
    {
        auto mod = anchor->getParentOfType<ModuleOp>();
        if (auto fn = mod.lookupSymbol<func::FuncOp>("knit_sample_term"))
            return fn;
        OpBuilder b(mod.getBodyRegion());
        auto i64 = b.getI64Type(), i1 = b.getI1Type();
        auto fnTy = b.getFunctionType({}, {i64, i1});
        auto fn = func::FuncOp::create(b, mod.getLoc(), "knit_sample_term", fnTy);
        fn.setPrivate();
        return fn;
    }

    // Which carry position carries the loop's boolean condition (the fail flag)?
    int failIndex(scf::WhileOp loop)
    {
        Block &before = loop.getBefore().front();
        auto cond = cast<scf::ConditionOp>(before.getTerminator());
        Value c = cond.getCondition();
        if (auto ba = dyn_cast<BlockArgument>(c))
            if (ba.getOwner() == &before)
                return ba.getArgNumber();
        if (auto ex = c.getDefiningOp<tensor::ExtractOp>())
            if (auto ba = dyn_cast<BlockArgument>(ex.getTensor()))
                if (ba.getOwner() == &before)
                    return ba.getArgNumber();
        return 0;
    }

    bool doRewrite(scf::WhileOp loop, int carryIdx, int C, Window win)
    {
        OpBuilder b(loop);
        Location loc = loop.getLoc();
        Type i32 = b.getI32Type(), f64 = b.getF64Type(), i1 = b.getI1Type();

        // applicability precheck: expval of a namedobs on the carried result
        ObsChain oc = findObsChain(loop, carryIdx);
        if (!oc.ok)
            return false;
        Type qT = oc.obs.getQubit().getType();
        func::FuncOp sampleFn = getOrCreateSampleFn(loop);
        int failIdx = failIndex(loop);
        unsigned nCarry = loop.getNumResults();

        // --- carry extension: +i32 counter, +f64 weight ---
        b.setInsertionPoint(loop);
        Value c0i32 = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
        Value fone = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(1.0));
        SmallVector<Value> inits(loop.getInits().begin(), loop.getInits().end());
        inits.push_back(c0i32);
        inits.push_back(fone);
        SmallVector<Type> resTys(loop.getResultTypes().begin(),
                                 loop.getResultTypes().end());
        resTys.push_back(i32);
        resTys.push_back(f64);
        auto nl = scf::WhileOp::create(b, loc, resTys, inits);

        // BEFORE region
        Block &oldBefore = loop.getBefore().front();
        Block *nb = b.createBlock(&nl.getBefore());
        for (Value in : inits)
            nb->addArgument(in.getType(), loc);
        IRMapping bmap;
        for (unsigned i = 0; i < oldBefore.getNumArguments(); ++i)
            bmap.map(oldBefore.getArgument(i), nb->getArgument(i));
        b.setInsertionPointToEnd(nb);
        for (Operation &op : oldBefore.without_terminator())
            b.clone(op, bmap);
        auto oldCond = cast<scf::ConditionOp>(oldBefore.getTerminator());
        SmallVector<Value> fwd;
        for (Value v : oldCond.getArgs())
            fwd.push_back(bmap.lookupOrDefault(v));
        unsigned n = oldBefore.getNumArguments();
        fwd.push_back(nb->getArgument(n));
        fwd.push_back(nb->getArgument(n + 1));
        scf::ConditionOp::create(b, loc, bmap.lookupOrDefault(oldCond.getCondition()),
                                   fwd);

        // AFTER region
        Block &oldAfter = loop.getAfter().front();
        Block *na = b.createBlock(&nl.getAfter());
        for (BlockArgument a : oldAfter.getArguments())
            na->addArgument(a.getType(), loc);
        Value itArg = na->addArgument(i32, loc);
        Value wArg = na->addArgument(f64, loc);
        IRMapping amap;
        for (unsigned i = 0; i < oldAfter.getNumArguments(); ++i)
            amap.map(oldAfter.getArgument(i), na->getArgument(i));
        b.setInsertionPointToEnd(na);
        for (Operation &op : oldAfter.without_terminator())
            b.clone(op, amap);
        auto oldYield = cast<scf::YieldOp>(oldAfter.getTerminator());

        // counter increment + periodic-cut guard
        Value c1i32 = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(1));
        Value it1 = arith::AddIOp::create(b, loc, itArg, c1i32);
        Value cC = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(C));
        Value rem = arith::RemSIOp::create(b, loc, it1, cC);
        Value zero = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
        Value atC =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rem, zero);
        // fail i1 (from the yielded value feeding the condition boolean)
        Value failVal = amap.lookupOrDefault(oldYield.getOperand(failIdx));
        Value failI1 = failVal;
        if (!failI1.getType().isInteger(1)) {
            if (isa<TensorType>(failI1.getType()))
                failI1 = tensor::ExtractOp::create(b, loc, failVal, ValueRange{});
        }
        Value docut = arith::AndIOp::create(b, loc, failI1, atC);

        Value carriedQ = amap.lookupOrDefault(oldYield.getOperand(carryIdx));
        auto cutIf = scf::IfOp::create(b, 
            loc, docut,
            [&](OpBuilder &tb, Location l) {
                auto pr = buildCut(tb, l, carriedQ, wArg, sampleFn);
                scf::YieldOp::create(tb, l, ValueRange{pr.first, pr.second});
            },
            [&](OpBuilder &eb, Location l) {
                scf::YieldOp::create(eb, l, ValueRange{carriedQ, wArg});
            });
        Value newQ = cutIf.getResult(0), newW = cutIf.getResult(1);

        SmallVector<Value> ny;
        for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
            Value v = amap.lookupOrDefault(oldYield.getOperand(i));
            ny.push_back(i == (unsigned)carryIdx ? newQ : v);
        }
        ny.push_back(it1);
        ny.push_back(newW);
        scf::YieldOp::create(b, loc, ny);

        // rewire original results, then drop the old loop
        for (unsigned i = 0; i < nCarry; ++i)
            loop.getResult(i).replaceAllUsesWith(nl.getResult(i));
        Value weightRes = nl.getResult(nCarry + 1);
        loop.erase();

        // --- output legalization (Part 3.4 step 4) ---
        b.setInsertionPoint(oc.obs);
        Value dq = oc.obs.getQubit();
        // capture the qubit's other consumers (e.g. a post-loop insert) up front
        SmallVector<OpOperand *> otherUses;
        for (OpOperand &u : dq.getUses())
            if (u.getOwner() != oc.obs.getOperation())
                otherUses.push_back(&u);
        Value pre = dq;
        if (oc.axis == 1) {
            pre = gate(b, loc, "Hadamard", dq);
        }
        else if (oc.axis == 2) {
            Value s = gate(b, loc, "S", dq);
            pre = gate(b, loc, "Hadamard", s);
        }
        auto mea = MeasureOp::create(b, loc, i1, qT, pre, IntegerAttr());
        Value z = mea.getMres(), dfin = mea.getOutQubit();
        Value fpos = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(1.0));
        Value fneg = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(-1.0));
        Value zv = arith::SelectOp::create(b, loc, z, fneg, fpos);
        Value e = arith::MulFOp::create(b, loc, weightRes, zv);
        oc.expval.getExpval().replaceAllUsesWith(e);
        // post-loop consumers of the carried qubit now use the measured-out qubit
        for (OpOperand *u : otherUses)
            u->set(dfin);
        oc.expval.erase();
        oc.obs.erase();

        nl->setAttr("knit.applied", b.getBoolAttr(true));
        nl->setAttr("knit.C", b.getI64IntegerAttr(C));
        nl->setAttr("knit.window", b.getDenseI64ArrayAttr({win.cMin, win.cMax}));
        nl->setAttr("knit.cut", b.getStringAttr("quasiprobability"));
        nl->setAttr("knit.strategy", b.getStringAttr("knit"));
        nl->setAttr("knit.known_state", b.getStringAttr("none"));
        return true;
    }

    // The cheap gamma = 1 cut (proven known carried state): extend the carry
    // with an i32 counter only, and at each cut deterministically measure the
    // carried wire (ending the coherent segment / clearing leakage) and
    // re-prepare the known state |psi0> (+ a counter-parity Pauli for the
    // KnownPauli case). No @knit_sample_term, no weight, no variance.
    bool doRewriteDeterministic(scf::WhileOp loop, int carryIdx, int C,
                                Window win, FrameResult fr, PrepChain prep)
    {
        OpBuilder b(loop);
        Location loc = loop.getLoc();
        Type i32 = b.getI32Type(), i1 = b.getI1Type();

        ObsChain oc = findObsChain(loop, carryIdx);
        if (!oc.ok)
            return false;
        Type qT = oc.obs.getQubit().getType();
        int failIdx = failIndex(loop);
        unsigned nCarry = loop.getNumResults();

        // replay |psi0>: prep gates on a fresh |0>
        auto reprep = [&](OpBuilder &tb, Location l, Value zeroQ) -> Value {
            Value q = zeroQ;
            for (const std::string &g : prep.gates)
                q = gate(tb, l, g, q);
            return q;
        };
        StringRef pName = fr.pauli == 'X'   ? "PauliX"
                          : fr.pauli == 'Y' ? "PauliY"
                                            : "PauliZ";
        bool knownPauli = fr.kind == Known::KnownPauli;

        // --- carry extension: +i32 counter only ---
        b.setInsertionPoint(loop);
        Value c0i32 = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
        SmallVector<Value> inits(loop.getInits().begin(), loop.getInits().end());
        inits.push_back(c0i32);
        SmallVector<Type> resTys(loop.getResultTypes().begin(),
                                 loop.getResultTypes().end());
        resTys.push_back(i32);
        auto nl = scf::WhileOp::create(b, loc, resTys, inits);

        // BEFORE region
        Block &oldBefore = loop.getBefore().front();
        Block *nb = b.createBlock(&nl.getBefore());
        for (Value in : inits)
            nb->addArgument(in.getType(), loc);
        IRMapping bmap;
        for (unsigned i = 0; i < oldBefore.getNumArguments(); ++i)
            bmap.map(oldBefore.getArgument(i), nb->getArgument(i));
        b.setInsertionPointToEnd(nb);
        for (Operation &op : oldBefore.without_terminator())
            b.clone(op, bmap);
        auto oldCond = cast<scf::ConditionOp>(oldBefore.getTerminator());
        SmallVector<Value> fwd;
        for (Value v : oldCond.getArgs())
            fwd.push_back(bmap.lookupOrDefault(v));
        unsigned n = oldBefore.getNumArguments();
        fwd.push_back(nb->getArgument(n));
        scf::ConditionOp::create(
            b, loc, bmap.lookupOrDefault(oldCond.getCondition()), fwd);

        // AFTER region
        Block &oldAfter = loop.getAfter().front();
        Block *na = b.createBlock(&nl.getAfter());
        for (BlockArgument a : oldAfter.getArguments())
            na->addArgument(a.getType(), loc);
        Value itArg = na->addArgument(i32, loc);
        IRMapping amap;
        for (unsigned i = 0; i < oldAfter.getNumArguments(); ++i)
            amap.map(oldAfter.getArgument(i), na->getArgument(i));
        b.setInsertionPointToEnd(na);
        for (Operation &op : oldAfter.without_terminator())
            b.clone(op, amap);
        auto oldYield = cast<scf::YieldOp>(oldAfter.getTerminator());

        Value c1i32 = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(1));
        Value it1 = arith::AddIOp::create(b, loc, itArg, c1i32);
        Value cC = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(C));
        Value rem = arith::RemSIOp::create(b, loc, it1, cC);
        Value zero = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
        Value atC =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rem, zero);
        Value failVal = amap.lookupOrDefault(oldYield.getOperand(failIdx));
        Value failI1 = failVal;
        if (!failI1.getType().isInteger(1) && isa<TensorType>(failI1.getType()))
            failI1 = tensor::ExtractOp::create(b, loc, failVal, ValueRange{});
        Value docut = arith::AndIOp::create(b, loc, failI1, atC);

        // counter parity (for KnownPauli): it1 & 1
        Value parity;
        if (knownPauli) {
            Value one = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(1));
            Value amask = arith::AndIOp::create(b, loc, it1, one);
            parity =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, amask, one);
        }

        Value carriedQ = amap.lookupOrDefault(oldYield.getOperand(carryIdx));
        auto cutIf = scf::IfOp::create(
            b, loc, docut,
            [&](OpBuilder &tb, Location l) {
                // measure to end the coherent segment, reset to |0>, re-prep
                auto m = MeasureOp::create(tb, l, i1, qT, carriedQ, IntegerAttr());
                Value q0 = guarded(tb, l, m.getMres(), m.getOutQubit(),
                                   [&](OpBuilder &gb, Value in) {
                                       return gate(gb, l, "PauliX", in);
                                   });
                Value psi = reprep(tb, l, q0);
                if (knownPauli)
                    psi = guarded(tb, l, parity, psi,
                                  [&](OpBuilder &gb, Value in) {
                                      return gate(gb, l, pName, in);
                                  });
                scf::YieldOp::create(tb, l, ValueRange{psi});
            },
            [&](OpBuilder &eb, Location l) {
                scf::YieldOp::create(eb, l, ValueRange{carriedQ});
            });
        Value newQ = cutIf.getResult(0);

        SmallVector<Value> ny;
        for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
            Value v = amap.lookupOrDefault(oldYield.getOperand(i));
            ny.push_back(i == (unsigned)carryIdx ? newQ : v);
        }
        ny.push_back(it1);
        scf::YieldOp::create(b, loc, ny);

        for (unsigned i = 0; i < nCarry; ++i)
            loop.getResult(i).replaceAllUsesWith(nl.getResult(i));
        Value counterRes = nl.getResult(nCarry); // i32 final trip count
        loop.erase();

        // --- output: REFRESH leaves the expval INTACT (Part 3.5). The carried
        // qubit's final state is a genuine quantum state (refreshed to the ideal
        // |psi0> each cut), so quantum.expval reads the right observable with no
        // weight and no legalization. For KnownPauli, correct the delivered
        // state by the counter parity before the observable.
        if (knownPauli) {
            b.setInsertionPoint(oc.obs);
            Value dq = oc.obs.getQubit();
            SmallVector<OpOperand *> uses; // namedobs + any post-loop insert
            for (OpOperand &u : dq.getUses())
                uses.push_back(&u);
            Value one = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(1));
            Value par = arith::AndIOp::create(b, loc, counterRes, one);
            Value pbit =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, par, one);
            StringRef pName = fr.pauli == 'X'   ? "PauliX"
                              : fr.pauli == 'Y' ? "PauliY"
                                                : "PauliZ";
            Value dqc = guarded(b, loc, pbit, dq,
                                [&](OpBuilder &gb, Value in) {
                                    return gate(gb, loc, pName, in);
                                });
            for (OpOperand *u : uses)
                u->set(dqc); // namedobs/insert use the corrected qubit
        }

        nl->setAttr("knit.applied", b.getBoolAttr(true));
        nl->setAttr("knit.C", b.getI64IntegerAttr(C));
        nl->setAttr("knit.window", b.getDenseI64ArrayAttr({win.cMin, win.cMax}));
        nl->setAttr("knit.cut", b.getStringAttr("deterministic"));
        nl->setAttr("knit.strategy", b.getStringAttr("refresh"));
        nl->setAttr("knit.known_state",
                    b.getStringAttr(knownPauli ? "pauli" : "identity"));
        return true;
    }
};

} // namespace quantum
} // namespace catalyst
