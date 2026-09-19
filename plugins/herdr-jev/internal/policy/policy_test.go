package policy

import "testing"

// These cases are a port of the reference implementation's tests/policy.test.ts. They
// encode the edge cases the asymmetric-bar policy must not get wrong, and porting them
// is the cheapest correctness win available — every one of them is a bug somebody
// already found.

func cfg() Config {
	return Config{
		Tiers:                  Tiers{Fast: "haiku", Balanced: "sonnet", Deep: "opus"},
		MinUpgradeConfidence:   0.3,
		MinDowngradeConfidence: 0.6,
		FamilyHints:            map[string]int{"haiku": 0, "sonnet": 1, "opus": 2},
	}
}

func f(v float64) *float64 { return &v }

// dec builds a decision the way a backend's answer reads.
func dec(tier Tier, conf float64, risky float64, effort float64, effortConf float64) *Decision {
	return &Decision{Tier: tier, Confidence: f(conf), Risky: f(risky), Effort: f(effort), EffortConfidence: f(effortConf)}
}

func TestEffortLevel(t *testing.T) {
	cases := []struct {
		score float64
		want  Effort
	}{
		{0, "low"}, {0.4, "low"}, {1.4, "medium"}, {2.04, "high"}, {3, "xhigh"}, {99, "xhigh"}, {-5, "low"},
	}
	for _, c := range cases {
		if got := EffortLevel(c.score); got != c.want {
			t.Errorf("EffortLevel(%v) = %q, want %q", c.score, got, c.want)
		}
	}
}

func TestAsymmetricBars(t *testing.T) {
	// 0.51 sits between the two bars: too low to downgrade, high enough to upgrade.
	// This single number is the whole design, so it gets its own test.
	cases := []struct {
		name    string
		d       *Decision
		current Current
		want    string
	}{
		{"a downgrade at 0.51 is refused", dec(TierFast, 0.51, 0.01, 1.4, 0.9), Current{Model: "claude-sonnet-5"}, ""},
		{"an upgrade at the same 0.51 is allowed", dec(TierDeep, 0.51, 0.01, 1.4, 0.9), Current{Model: "claude-sonnet-5"}, "opus"},
		{"a confident downgrade is allowed", dec(TierFast, 0.95, 0.01, 1.4, 0.9), Current{Model: "claude-opus-5"}, "haiku"},
		{"a confident upgrade is allowed", dec(TierDeep, 0.9, 0.01, 1.4, 0.9), Current{Model: "claude-haiku-4-5"}, "opus"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Route(c.d, c.current, cfg()).Model; got != c.want {
				t.Errorf("model = %q, want %q", got, c.want)
			}
		})
	}
}

func TestEffortMovesBothWays(t *testing.T) {
	cases := []struct {
		name    string
		d       *Decision
		current Current
		want    Effort
	}{
		{"down to low", dec(TierFast, 0.95, 0.01, 0.1, 0.9), Current{Model: "claude-haiku-4-5", Effort: "high"}, "low"},
		{"up to xhigh", dec(TierDeep, 0.95, 0.01, 2.8, 0.9), Current{Model: "claude-opus-5", Effort: "low"}, "xhigh"},
		{"a numeric effort is the caller's own scale and is left alone",
			dec(TierDeep, 0.95, 0.01, 2.8, 0.9), Current{Model: "claude-opus-5", Effort: 4000.0}, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Route(c.d, c.current, cfg()).Effort; got != c.want {
				t.Errorf("effort = %q, want %q", got, c.want)
			}
		})
	}
}

func TestRiskForcesDeepPastBothBars(t *testing.T) {
	// A mechanically simple task that nonetheless does something final. The tier
	// answer is 0.97 for FAST, which would otherwise be an easy, confident downgrade.
	d := dec(TierFast, 0.97, 0.93, 1.4, 0.9)
	r := Route(d, Current{Model: "claude-sonnet-5", Effort: "low"}, cfg())
	if r.Model != "opus" {
		t.Errorf("model = %q, want opus", r.Model)
	}
	if r.Effort != "high" {
		t.Errorf("effort = %q, want high", r.Effort)
	}
	if !contains(r.Reason, "risk") {
		t.Errorf("reason %q should name risk", r.Reason)
	}
}

func TestRiskRaisesTheEffortFloorButNeverLowersOne(t *testing.T) {
	// A prod deletion: risky, but mechanically simple, so its own effort is 0.
	// Forcing skips the thresholds, so without the clamp this pulls xhigh DOWN to
	// high with no confidence check at all — the opposite of what the rule is for.
	d := dec(TierFast, 0.99, 0.95, 0, 0.9)
	cases := []struct {
		current any
		want    Effort
	}{
		{"xhigh", ""},
		{"max", ""},
		{"low", "high"},
	}
	for _, c := range cases {
		got := Route(d, Current{Model: "claude-opus-5", Effort: c.current}, cfg()).Effort
		if got != c.want {
			t.Errorf("from %v: effort = %q, want %q", c.current, got, c.want)
		}
	}
}

func TestMaxRanksAboveEveryRubricRung(t *testing.T) {
	maxRank, xhighRank := EffortRank("max"), EffortRank("xhigh")
	if maxRank == nil || xhighRank == nil || *maxRank <= *xhighRank {
		t.Fatalf("max must rank above xhigh, got %v vs %v", maxRank, xhighRank)
	}
	// Leaving max is a DOWNGRADE, so it needs the high bar, not the lenient one.
	barely := &Decision{Tier: TierFast, Confidence: f(0.9), Risky: f(0.01), Effort: f(0), EffortConfidence: f(0.35)}
	if got := Route(barely, Current{Model: "claude-sonnet-5", Effort: "max"}, cfg()).Effort; got != "" {
		t.Errorf("a 0.35-confidence downgrade from max should be refused, got %q", got)
	}
	sure := &Decision{Tier: TierFast, Confidence: f(0.9), Risky: f(0.01), Effort: f(0), EffortConfidence: f(0.8)}
	if got := Route(sure, Current{Model: "claude-sonnet-5", Effort: "max"}, cfg()).Effort; got != "low" {
		t.Errorf("a 0.8-confidence downgrade from max should be allowed, got %q", got)
	}
}

func TestMissingConfidenceMayOnlyMoveUp(t *testing.T) {
	// Spending less on an unmeasured hunch is the bad trade.
	down := &Decision{Tier: TierFast}
	if got := Route(down, Current{Model: "claude-opus-5"}, cfg()).Model; got != "" {
		t.Errorf("a confidence-less downgrade should be refused, got %q", got)
	}
	up := &Decision{Tier: TierDeep}
	if got := Route(up, Current{Model: "claude-haiku-4-5"}, cfg()).Model; got != "opus" {
		t.Errorf("a confidence-less upgrade should be allowed, got %q", got)
	}
}

func TestUnrecognisedModelIsTreatedAsAnUpgradeNotGuessedAt(t *testing.T) {
	if r := RankOf("some-other-vendor-model", cfg().Tiers, cfg().FamilyHints); r != nil {
		t.Fatalf("expected no rank, got %v", *r)
	}
	// 0.35 clears the upgrade bar only; an undecidable direction gets that one.
	d := dec(TierFast, 0.35, 0.01, 1.4, 0.9)
	if got := Route(d, Current{Model: "some-other-vendor-model"}, cfg()).Model; got != "haiku" {
		t.Errorf("model = %q, want haiku", got)
	}
}

func TestRankOfPrefersTheLongestFamilyHint(t *testing.T) {
	// "gpt-5-mini" must not rank as "gpt-5". A shorter hint winning here would rank
	// the cheap model as the balanced one and quietly disable every downgrade.
	tiers := Tiers{Fast: "", Balanced: "", Deep: ""}
	hints := map[string]int{"gpt-5": 1, "gpt-5-mini": 0}
	got := RankOf("gpt-5-mini", tiers, hints)
	if got == nil || *got != 0 {
		t.Fatalf("gpt-5-mini should rank 0, got %v", got)
	}
}

func TestNoChangeAndNoDecisionAreDifferentAndBothSaySo(t *testing.T) {
	// A router that classified and left the request alone must be distinguishable
	// from one that never ran. Otherwise silence means nothing.
	if got := Route(nil, Current{Model: "sonnet", Effort: "medium"}, cfg()).Reason; got != "no decision" {
		t.Errorf("reason = %q, want %q", got, "no decision")
	}
	d := &Decision{Tier: TierFast, Confidence: f(0.41), Effort: f(0), EffortConfidence: f(0.41)}
	r := Route(d, Current{Model: "sonnet", Effort: "medium"}, cfg())
	if r.Changed() {
		t.Fatalf("expected no change, got %+v", r)
	}
	if want := "kept sonnet/medium, wanted haiku/low (confidence 0.41)"; r.Reason != want {
		t.Errorf("reason = %q, want %q", r.Reason, want)
	}

	noConf := &Decision{Tier: TierFast, Effort: f(0)}
	if want := "kept sonnet/medium, wanted haiku/low (confidence n/d)"; Route(noConf, Current{Model: "sonnet", Effort: "medium"}, cfg()).Reason != want {
		t.Errorf("a confidence-less no-change must say n/d rather than going quiet")
	}
}

func TestSameTierKeepsTheExactModelId(t *testing.T) {
	// A session already on the deep tier's model keeps its exact id — including a
	// variant like an extended-context build that the tier name does not name.
	d := dec(TierDeep, 0.95, 0.01, 1.4, 0.9)
	if got := Route(d, Current{Model: "claude-opus-5[1m]", Effort: "medium"}, cfg()).Model; got != "" {
		t.Errorf("model = %q, want no change", got)
	}
}

func TestDescribeDecisionCarriesEveryAnswerAndTheLatency(t *testing.T) {
	d := &Decision{Tier: TierDeep, Confidence: f(0.95), Effort: f(2.8), EffortConfidence: f(0.90), Risky: f(0.01)}
	want := "tier deep (0.95) · effort 2.8 → xhigh (0.90) · risky 0.01 · 249ms"
	if got := DescribeDecision(d, 249.4); got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
	if got := DescribeDecision(nil, 800); got != "no answer · 800ms" {
		t.Errorf("a classification that never answered must say so, got %q", got)
	}
}

func TestDescribeStatusDistinguishesAChangeFromADeliberateNoChange(t *testing.T) {
	d := &Decision{Tier: TierFast, Confidence: f(0.87)}
	if got := DescribeStatus(d, Routing{Model: "haiku", Effort: "low"}); got != "jev · fast 0.87 → haiku/low" {
		t.Errorf("got %q", got)
	}
	if got := DescribeStatus(d, Routing{}); got != "jev · fast 0.87 · unchanged" {
		t.Errorf("got %q", got)
	}
	if got := DescribeStatus(nil, Routing{}); got != "jev · no answer" {
		t.Errorf("got %q", got)
	}
}

func TestDescribeSetupNamesTheBackendAndWhatIsOn(t *testing.T) {
	got := DescribeSetup("openjev", "http://127.0.0.1:21131", []string{"model", "effort"})
	if want := "ready on openjev (http://127.0.0.1:21131); routing model, effort"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if got := DescribeSetup("nothing", "", nil); got != "ready on nothing; routing nothing, every switch is off" {
		t.Errorf("got %q", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
