// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap.
#include "strata/spec/draft_policy.hpp"

#include <cstdio>

using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    // --spec-cost: the MTP window from calibrated draft probabilities and the measured cost per size
    {
        // a DDR4 PC: ~10 ms per window token; drafts that claim 0.6 are accepted 30% of the time, 0.95 ones 95%
        DraftPolicy p(4);
        const float lo[3] = {0.6f, 0.6f, 0.6f}, hi[3] = {0.95f, 0.95f, 0.95f};
        for (int i = 0; i < 400; ++i) {
            const bool ok = (i % 10) < 3;
            p.observe_mtp_drafts(lo, 2, ok ? 1 : 0);
            p.observe_mtp_drafts(hi, 4, (i % 20) == 0 ? 2 : 3);
            p.observe(false, 2, ok ? 1 : 0, 0, cost(2));
            p.observe(false, 1, 0, 0, cost(1));
            p.observe(false, 3, 2, 0, cost(3));
            p.observe(false, 4, 3, 0, cost(4));
        }
        check(p.draft_rate(0.6f) < 0.4 && p.draft_rate(0.95f) > 0.85, "calibration: 0.6 drafts ~0.3, 0.95 drafts ~0.95");
        int full = 0, cut = 0;
        for (int i = 0; i < 64; ++i) {
            if (p.choose_mtp(lo, 3) == 1) ++cut;
            if (p.choose_mtp(hi, 3) == 4) ++full;
        }
        check(cut >= 56, "overconfident, costly drafts: the window shrinks to the token alone");
        check(full == 64, "reliable drafts: the whole window is kept");
        check(p.choose_mtp(hi, 0) == 1, "no drafts: a window of one");
    }
    {
        DraftPolicy p(4);
        const float lo[3] = {0.6f, 0.6f, 0.6f};
        check(p.choose_mtp(lo, 3) == 4, "before any round is measured: the floor's window");
        for (int i = 0; i < 100; ++i) {
            p.observe_mtp_drafts(lo, 4, 0);
            p.observe(false, 4, 0, 0, cost(4));
            p.observe(false, 1, 0, 0, cost(1));
        }
        int probes = 0;
        for (int i = 0; i < 64; ++i) probes += p.choose_mtp(lo, 3) == 4;
        check(probes == 64 / DraftPolicy::kProbeEvery, "every kProbeEvery-th call still measures the floor's window");
        check(p.choose_mtp(lo, 9) <= 4, "never beyond the window cap");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
