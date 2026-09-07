// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

// QSimQubit -- a Catalyst runtime device (QuantumDevice plugin) that is the C++
// port of mlir/purl/sim/qsim.py: a Monte-Carlo TRAJECTORY statevector simulator
// with the Purl noise model (per-gate depolarizing, idle amplitude-damping +
// pure-dephasing over gate/readout/feedback durations, per-2q-gate ABSORBING
// leakage, readout flip). It lets a compiled Catalyst program -- with the two Purl
// passes in the pipeline -- execute against the SAME hardware calibration the
// passes read, so delivered fidelity under decoherence is measured end to end.
//
// The flat calibration (T1/T2, gate/readout/tau durations, p1/p2/p_ro/p_meas,
// p_leak) and the global noise scale `lam` arrive as device kwargs; the Python
// device (frontend) flattens them from the shared ibm_eagle_r3.json via the same
// ibm_dataset.carried_calib loader the passes use. One trajectory per execution;
// the eval harness averages over calls (as sim/qsim.py is used per shot).

#include <cmath>
#include <complex>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataView.hpp"
#include "QuantumDevice.hpp"
#include "Types.h"
#include "Utils.hpp"

namespace Catalyst::Runtime::Devices {

using cd = std::complex<double>;

class QSimQubit final : public Catalyst::Runtime::QuantumDevice {
    // --- calibration (flat, kwargs; SI seconds / probabilities) ---
    double gate1q_ = 30e-9, gate2q_ = 60e-9, readout_ = 700e-9, tau_ = 500e-9;
    double T1_ = INFINITY, T2_ = INFINITY;
    double p1_ = 0.0, p2_ = 0.0, p_ro_ = 0.0, p_meas_ = 0.0, p_leak_ = 0.0;
    double lam_ = 1.0;                 // global noise scale (0 = exactly noiseless)
    double inv_T1_ = 0.0, inv_Tphi_ = 0.0;

    // --- trajectory state ---
    std::size_t n_ = 0, shots_ = 0;
    std::vector<cd> psi_;             // statevector, little-endian (qubit 0 = LSB)
    std::vector<char> leaked_;        // absorbing per-qubit leakage flags
    std::mt19937 own_rng_{0};
    std::mt19937 *rng_ = &own_rng_;   // runtime PRNG when provided (reproducible)

    // measurement-result storage (Result = bool*), plus a cached observable
    bool res_true_ = true, res_false_ = false;
    struct Obs { ObsId id; QubitIdType wire; };
    std::vector<Obs> obs_;

    double urand() { return std::uniform_real_distribution<double>(0.0, 1.0)(*rng_); }

  public:
    explicit QSimQubit(const std::string &kwargs = "{}")
    {
        auto kv = Catalyst::Runtime::parse_kwargs(kwargs);
        auto get = [&](const char *k, double dflt) {
            auto it = kv.find(k);
            return it == kv.end() ? dflt : std::stod(it->second);
        };
        gate1q_ = get("gate_1q", gate1q_);
        gate2q_ = get("gate_2q", gate2q_);
        readout_ = get("readout", readout_);
        tau_ = get("tau", tau_);
        T1_ = get("T1", T1_);
        T2_ = get("T2", T2_);
        p1_ = get("p1", 0.0);
        p2_ = get("p2", 0.0);
        p_ro_ = get("p_ro", 0.0);
        p_meas_ = get("p_meas", 0.0);
        p_leak_ = get("p_leak", 0.0);
        lam_ = get("lam", 1.0);
        if (!std::isinf(T1_) && !std::isinf(T2_) && T1_ > 0 && T2_ > 0) {
            inv_T1_ = 1.0 / T1_;
            inv_Tphi_ = std::max(0.0, 1.0 / T2_ - 1.0 / (2.0 * T1_));
        }
    }
    ~QSimQubit() override = default;
    QSimQubit(const QSimQubit &) = delete;
    QSimQubit &operator=(const QSimQubit &) = delete;
    QSimQubit(QSimQubit &&) = delete;
    QSimQubit &operator=(QSimQubit &&) = delete;

    // ---------------- qubit / execution management ----------------
    auto AllocateQubits(std::size_t n) -> std::vector<QubitIdType> override
    {
        n_ = n;
        psi_.assign(std::size_t(1) << n_, cd(0.0, 0.0));
        if (!psi_.empty())
            psi_[0] = cd(1.0, 0.0);
        leaked_.assign(n_, 0);
        std::vector<QubitIdType> ids(n_);
        for (std::size_t i = 0; i < n_; ++i)
            ids[i] = (QubitIdType)i;
        return ids;
    }
    void ReleaseQubits(const std::vector<QubitIdType> &) override
    {
        psi_.clear();
        leaked_.clear();
        n_ = 0;
    }
    auto GetNumQubits() const -> std::size_t override { return n_; }
    void SetDeviceShots(std::size_t s) override { shots_ = s; }
    auto GetDeviceShots() const -> std::size_t override { return shots_; }
    void SetDevicePRNG(std::mt19937 *gen) override
    {
        if (gen)
            rng_ = gen;
    }

  private:
    // ---------------- statevector kernels (little-endian) ----------------
    static std::size_t bit(std::size_t idx, std::size_t q) { return (idx >> q) & 1U; }

    void apply1q(const cd U[2][2], std::size_t q)
    {
        const std::size_t step = std::size_t(1) << q;
        for (std::size_t i = 0; i < psi_.size(); ++i) {
            if (bit(i, q))
                continue;
            std::size_t j = i | step;
            cd a = psi_[i], b = psi_[j];
            psi_[i] = U[0][0] * a + U[0][1] * b;
            psi_[j] = U[1][0] * a + U[1][1] * b;
        }
    }

    // apply the 2x2 U on `target`, conditioned on every control bit being 1
    void applyCtrl1q(const cd U[2][2], const std::vector<std::size_t> &controls,
                     std::size_t target)
    {
        const std::size_t step = std::size_t(1) << target;
        for (std::size_t i = 0; i < psi_.size(); ++i) {
            if (bit(i, target))
                continue;
            bool ok = true;
            for (auto c : controls)
                if (!bit(i, c)) {
                    ok = false;
                    break;
                }
            if (!ok)
                continue;
            std::size_t j = i | step;
            cd a = psi_[i], b = psi_[j];
            psi_[i] = U[0][0] * a + U[0][1] * b;
            psi_[j] = U[1][0] * a + U[1][1] * b;
        }
    }

    double probOne(std::size_t q) const
    {
        double p = 0.0;
        for (std::size_t i = 0; i < psi_.size(); ++i)
            if (bit(i, q))
                p += std::norm(psi_[i]);
        return p;
    }

    void collapse(std::size_t q, int outcome)
    {
        double nrm = 0.0;
        for (std::size_t i = 0; i < psi_.size(); ++i) {
            if ((int)bit(i, q) != outcome)
                psi_[i] = cd(0.0, 0.0);
            else
                nrm += std::norm(psi_[i]);
        }
        if (nrm > 0) {
            double s = 1.0 / std::sqrt(nrm);
            for (auto &z : psi_)
                z *= s;
        }
    }

    // ---------------- noise channels (port of qsim.py) ----------------
    static void pauli(char kind, cd U[2][2])
    {
        if (kind == 'X') { U[0][0] = 0; U[0][1] = 1; U[1][0] = 1; U[1][1] = 0; }
        else if (kind == 'Y') { U[0][0] = 0; U[0][1] = cd(0,-1); U[1][0] = cd(0,1); U[1][1] = 0; }
        else { U[0][0] = 1; U[0][1] = 0; U[1][0] = 0; U[1][1] = -1; }  // Z
    }

    void applyPauli(char kind, std::size_t q)
    {
        cd U[2][2];
        pauli(kind, U);
        apply1q(U, q);
    }

    // trajectory idle: amplitude damping + pure dephasing on qubit q for time dt
    void idleQubit(std::size_t q, double dt)
    {
        if (dt <= 0 || lam_ == 0.0)
            return;
        double eff = lam_ * dt;
        if (inv_T1_ > 0) {
            double gamma = 1.0 - std::exp(-eff * inv_T1_);
            if (gamma > 0) {
                double p1 = probOne(q);
                const std::size_t step = std::size_t(1) << q;
                if (urand() < gamma * p1) {
                    // quantum jump |0><1|: move population 1 -> 0, kill 1-branch
                    for (std::size_t i = 0; i < psi_.size(); ++i)
                        if (!bit(i, q)) {
                            psi_[i] = psi_[i | step];
                            psi_[i | step] = cd(0.0, 0.0);
                        }
                }
                else {
                    double s = std::sqrt(1.0 - gamma);
                    for (std::size_t i = 0; i < psi_.size(); ++i)
                        if (bit(i, q))
                            psi_[i] *= s;
                }
                double nrm = 0.0;
                for (auto &z : psi_)
                    nrm += std::norm(z);
                if (nrm > 0) {
                    double inv = 1.0 / std::sqrt(nrm);
                    for (auto &z : psi_)
                        z *= inv;
                }
            }
        }
        if (inv_Tphi_ > 0) {
            double p_pd = 1.0 - std::exp(-eff * inv_Tphi_);
            if (urand() < p_pd / 2.0)
                applyPauli('Z', q);
        }
    }

    void idleOthers(const std::vector<std::size_t> &active, double dt)
    {
        if (dt <= 0 || lam_ == 0.0)
            return;
        for (std::size_t q = 0; q < n_; ++q) {
            bool act = false;
            for (auto a : active)
                if (a == q) { act = true; break; }
            if (!act)
                idleQubit(q, dt);
        }
    }

    // per-gate depolarizing on the operated qubits (uniform Pauli, joint for nq>1)
    void depol(const std::vector<std::size_t> &qs)
    {
        if (lam_ == 0.0)
            return;
        std::size_t k = qs.size();
        double p = (k >= 2) ? p2_ : p1_;
        if (p <= 0.0)
            return;
        if (urand() >= lam_ * p)
            return;
        if (k == 1) {
            int r = std::uniform_int_distribution<int>(0, 2)(*rng_);
            applyPauli(r == 0 ? 'X' : r == 1 ? 'Y' : 'Z', qs[0]);
            return;
        }
        // joint k-qubit depolarizing: a uniform non-identity Pauli string
        std::vector<int> picks(k);
        bool any = false;
        while (!any)
            for (std::size_t i = 0; i < k; ++i) {
                picks[i] = std::uniform_int_distribution<int>(0, 3)(*rng_);
                any = any || picks[i] != 0;
            }
        for (std::size_t i = 0; i < k; ++i)
            if (picks[i] == 1) applyPauli('X', qs[i]);
            else if (picks[i] == 2) applyPauli('Y', qs[i]);
            else if (picks[i] == 3) applyPauli('Z', qs[i]);
    }

    // per-2q-gate ABSORBING leakage on every qubit the gate touches
    void leak2q(const std::vector<std::size_t> &qs)
    {
        if (lam_ == 0.0 || p_leak_ <= 0.0)
            return;
        double pl = lam_ * p_leak_;
        for (auto q : qs)
            if (!leaked_[q] && urand() < pl)
                leaked_[q] = 1;
    }

    bool anyLeaked(const std::vector<std::size_t> &qs) const
    {
        for (auto q : qs)
            if (leaked_[q])
                return true;
        return false;
    }

    // build the 2x2 unitary of a 1q named gate (inverse-aware)
    static bool unitary1q(const std::string &name, const std::vector<double> &p,
                          bool inv, cd U[2][2])
    {
        const double s2 = 1.0 / std::sqrt(2.0);
        if (name == "Hadamard") { U[0][0]=s2; U[0][1]=s2; U[1][0]=s2; U[1][1]=-s2; return true; }
        if (name == "PauliX") { pauli('X', U); return true; }
        if (name == "PauliY") { pauli('Y', U); return true; }
        if (name == "PauliZ") { pauli('Z', U); return true; }
        if (name == "Identity") { U[0][0]=1; U[0][1]=0; U[1][0]=0; U[1][1]=1; return true; }
        if (name == "S") { U[0][0]=1; U[0][1]=0; U[1][0]=0; U[1][1]= inv?cd(0,-1):cd(0,1); return true; }
        if (name == "T") { double a=M_PI/4*(inv?-1:1); U[0][0]=1; U[0][1]=0; U[1][0]=0; U[1][1]=std::exp(cd(0,a)); return true; }
        if (name == "RX") { double a=(inv?-1:1)*p[0]/2; U[0][0]=std::cos(a); U[0][1]=cd(0,-std::sin(a)); U[1][0]=cd(0,-std::sin(a)); U[1][1]=std::cos(a); return true; }
        if (name == "RY") { double a=(inv?-1:1)*p[0]/2; U[0][0]=std::cos(a); U[0][1]=-std::sin(a); U[1][0]=std::sin(a); U[1][1]=std::cos(a); return true; }
        if (name == "RZ") { double a=(inv?-1:1)*p[0]/2; U[0][0]=std::exp(cd(0,-a)); U[0][1]=0; U[1][0]=0; U[1][1]=std::exp(cd(0,a)); return true; }
        if (name == "PhaseShift") { double a=(inv?-1:1)*p[0]; U[0][0]=1; U[0][1]=0; U[1][0]=0; U[1][1]=std::exp(cd(0,a)); return true; }
        return false;
    }

  public:
    // ---------------- gate application ----------------
    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse = false,
                        const std::vector<QubitIdType> &controlled_wires = {},
                        const std::vector<bool> & = {},
                        const std::vector<std::string> & = {}) override
    {
        std::vector<std::size_t> w;
        for (auto q : wires)
            w.push_back((std::size_t)q);
        std::vector<std::size_t> ctrl;
        for (auto q : controlled_wires)
            ctrl.push_back((std::size_t)q);

        // multi-qubit named gates -> canonicalize to (controls..., target)
        std::string nm = name;
        std::vector<double> pr = params;
        if (nm == "CNOT") { nm = "PauliX"; ctrl.push_back(w[0]); w = {w[1]}; }
        else if (nm == "CZ") { nm = "PauliZ"; ctrl.push_back(w[0]); w = {w[1]}; }
        else if (nm == "Toffoli") { nm = "PauliX"; ctrl.push_back(w[0]); ctrl.push_back(w[1]); w = {w[2]}; }
        else if (nm == "CRX") { nm = "RX"; ctrl.push_back(w[0]); w = {w[1]}; }
        else if (nm == "CRY") { nm = "RY"; ctrl.push_back(w[0]); w = {w[1]}; }
        else if (nm == "CRZ") { nm = "RZ"; ctrl.push_back(w[0]); w = {w[1]}; }

        std::size_t arity = w.size() + ctrl.size();
        std::vector<std::size_t> touched = w;
        touched.insert(touched.end(), ctrl.begin(), ctrl.end());

        // idle spectators for this op's duration (arity-based), like qsim
        idleOthers(touched, arity >= 2 ? gate2q_ : gate1q_);

        // absorbing-leakage no-op: a 2q gate touching a leaked wire does not act
        if (!(arity >= 2 && anyLeaked(touched))) {
            cd U[2][2];
            if (!unitary1q(nm, pr, inverse, U))
                RT_FAIL(("QSimQubit: unsupported gate " + name).c_str());
            if (ctrl.empty())
                apply1q(U, w[0]);
            else
                applyCtrl1q(U, ctrl, w[0]);
            depol(touched);
        }
        if (arity >= 2)
            leak2q(touched);
    }

    auto Measure(QubitIdType wire, std::optional<int32_t> postselect) -> Result override
    {
        std::size_t q = (std::size_t)wire;
        idleOthers({q}, readout_);
        // a leaked qubit reads out as garbage (uniform bit), no collapse
        if (leaked_[q]) {
            bool g = (urand() < 0.5);
            return g ? &res_true_ : &res_false_;
        }
        if (lam_ > 0 && p_meas_ > 0 && urand() < lam_ * p_meas_) {
            int r = std::uniform_int_distribution<int>(0, 2)(*rng_);
            applyPauli(r == 0 ? 'X' : r == 1 ? 'Y' : 'Z', q);
        }
        int outcome;
        if (postselect.has_value())
            outcome = *postselect;
        else
            outcome = (urand() < probOne(q)) ? 1 : 0;
        collapse(q, outcome);
        int reported = outcome;
        if (lam_ > 0 && p_ro_ > 0 && urand() < lam_ * p_ro_)
            reported = 1 - outcome;
        return reported ? &res_true_ : &res_false_;
    }

    // ---------------- observables + expval ----------------
    auto Observable(ObsId id, const std::vector<cd> &,
                    const std::vector<QubitIdType> &wires) -> ObsIdType override
    {
        obs_.push_back({id, wires.empty() ? 0 : wires[0]});
        return (ObsIdType)(obs_.size() - 1);
    }

    auto Expval(ObsIdType key) -> double override
    {
        const Obs &o = obs_[(std::size_t)key];
        std::size_t q = (std::size_t)o.wire;
        if (o.id == ObsId::Identity)
            return 1.0;
        // a leaked wire is out of the computational subspace -> analytic <P> = 0
        if (q < leaked_.size() && leaked_[q])
            return 0.0;
        if (o.id == ObsId::PauliZ) {
            double e = 0.0;
            const std::size_t step = std::size_t(1) << q;
            for (std::size_t i = 0; i < psi_.size(); ++i)
                e += (i & step ? -1.0 : 1.0) * std::norm(psi_[i]);
            return e;
        }
        // PauliX / PauliY: <P> = <psi| P_q |psi> via the flipped-partner amplitude
        cd acc(0.0, 0.0);
        const std::size_t step = std::size_t(1) << q;
        for (std::size_t i = 0; i < psi_.size(); ++i) {
            if (bit(i, q))
                continue;
            cd a = psi_[i], b = psi_[i | step];
            if (o.id == ObsId::PauliX)
                acc += std::conj(a) * b + std::conj(b) * a;
            else // PauliY: |0><1| -i, |1><0| +i
                acc += std::conj(a) * (cd(0, -1) * b) + std::conj(b) * (cd(0, 1) * a);
        }
        return acc.real();
    }
};

} // namespace Catalyst::Runtime::Devices
