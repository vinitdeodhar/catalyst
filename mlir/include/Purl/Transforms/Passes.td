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

#ifndef PURL_PASSES
#define PURL_PASSES

include "mlir/Pass/PassBase.td"

def PurlPass : Pass<"purl"> {
    let summary = "Detect carry-type dynamic loops, select a cut strategy from the "
                  "hardware cost model, and emit an abstract purl.qcut at each cut site.";
    let description = [{
        The analysis + rewrite half of Purl (spec 3). Classifies the loop, computes
        body depth, proves the carried state (for refresh), runs the profitability
        cost model, and -- when profitable -- emits a single purl.qcut op inside the
        periodic cut guard with the chosen strategy baked in. Emits purl.* attributes
        (class, window, known_state, strategy, C, predicted, predicted_fidelity).
    }];

    let dependentDialects = [
        "arith::ArithDialect",
        "scf::SCFDialect",
        "tensor::TensorDialect",
        "func::FuncDialect",
        "catalyst::quantum::QuantumDialect",
        "catalyst::purl::PurlDialect"
    ];

    let options = [
        Option<"calib", "calib", "std::string", /*default=*/"\"\"",
            "Path to the hardware calibration JSON (or 'unit' for layer counting).">,
        Option<"pSuccess", "p", "double", /*default=*/"0.625",
            "RUS per-iteration success probability (profile input).">,
        Option<"cutPeriod", "C", "int", /*default=*/"0",
            "Override the cut period C (0 = cost-model arg-min).">,
        Option<"budgetFraction", "f", "double", /*default=*/"0.05",
            "Coherence-budget fraction for C_max = floor(f*T2/(B+tau)).">,
        Option<"analyzeOnly", "analyze-only", "bool", /*default=*/"false",
            "Emit purl.* attributes without rewriting the IR.">,
        Option<"shots", "shots", "int", /*default=*/"0",
            "Shot budget S (0 = legacy: fire best mechanism, no profitability veto).">,
        Option<"margin", "margin", "double", /*default=*/"1.0",
            "Profitability margin: fire iff predicted*margin < predicted(NONE).">,
        Option<"sigma0", "sigma0", "double", /*default=*/"1.0",
            "Per-shot standard deviation of the observable (profile input).">,
        Option<"carryQubit", "carry-qubit", "int", /*default=*/"0",
            "Physical qubit index the carried wire maps to (placement input).">,
        Option<"depth", "depth", "int", /*default=*/"0",
            "Report purl.fidelity_at_depth at this runtime depth D in iterations (0 = off).">
    ];
}

def LowerQCutPass : Pass<"purl-lower-qcut"> {
    let summary = "Mechanically expand each purl.qcut into its concrete op sequence.";
    let description = [{
        The code-generation half of Purl (spec 3.7). A purely mechanical rewrite that
        switches on purl.qcut's `strategy`: `refresh` -> measure + reset + inline the
        `prep` region (+ pauli_correction); `knit` -> sample-term RNG hook + basis
        change to `axis` + measure + reset + eigenstate prep + 4*sigma*s weight fold.
        Performs NO analysis; reads only the op.
    }];

    let dependentDialects = [
        "arith::ArithDialect",
        "scf::SCFDialect",
        "func::FuncDialect",
        "catalyst::quantum::QuantumDialect",
        "catalyst::purl::PurlDialect"
    ];
}

#endif // PURL_PASSES
