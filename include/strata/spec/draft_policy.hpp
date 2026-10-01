// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
//
// The MTP window's own length (`choose_mtp`, opt-in with --spec-cost): a fixed draft-probability floor (--spec-min-p)
// keeps a draft whatever it costs to verify, and the MTP's probabilities are not acceptance rates - on a DDR4 PC
// whose missed experts stream from RAM, drafts of probability 0.5-0.9 were accepted about a third of the time while
// each window token added ~10 ms of CPU expert work. So each draft's probability is calibrated against what
// verification accepted (decayed counts per probability bucket), and the window is the size T maximising
//   E(T) - lambda * cost(T),   E(T) = 1 + sum_{i<T} prod_{m<=i} q(p_m),   lambda = recent committed tokens per ms
// (the rate-maximising rule for a renewal process), with the same measured costs per size.
#pragma once

#include <array>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens

    /// --spec-cost: the MTP window (1 + drafts) for drafts of probabilities `probs[0..n_drafts)`, which the caller
    /// has already cut at its probability floor. Until enough rounds are measured, and on every `kProbeEvery`-th
    /// call (so the larger sizes keep being measured), it returns the floor's window, `n_drafts + 1`.
    int choose_mtp(const float* probs, int n_drafts);
    /// After every MTP window (with or without --spec-cost): its t - 1 drafts' probabilities and how many were
    /// accepted. Drafts after the first rejected one were never checked and are not counted.
    void observe_mtp_drafts(const float* probs, int t, int accepted);
    double draft_rate(float p) const;      // calibrated acceptance of a draft of probability p
    double throughput() const;             // committed tokens per ms over the recent rounds (0 = not measured yet)
    static constexpr int kPBins = 10;
    static constexpr int kProbeEvery = 16;

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;

    int max_t_;
    double margin_;
    std::array<double, kMaxT + 1> cost_{}, cost_n_{};      // round ms by window size
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kPBins> p_ok_{}, p_n_{};            // MTP drafts accepted / checked, by probability bucket
    double rate_tok_ = 0.0, rate_ms_ = 0.0;                // EMA of committed tokens and of round ms
    int rounds_ = 0;
    unsigned calls_ = 0;
};

}  // namespace strata::spec
