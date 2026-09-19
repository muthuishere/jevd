package classify

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/muthuishere/herdr-jev/internal/jev"
	"github.com/muthuishere/herdr-jev/internal/policy"
)

// fake answers by option count: three options is the tier question, four is the effort
// rubric. That is enough to drive every branch without a model.
type fake struct {
	tier   []float64
	effort []float64
	risky  float64
	err    error
	tierErr error
}

func (f fake) Rerank(_ context.Context, _ string, options []string) ([]jev.Scored, error) {
	if f.err != nil {
		return nil, f.err
	}
	src := f.effort
	if len(options) == 3 {
		if f.tierErr != nil {
			return nil, f.tierErr
		}
		src = f.tier
	}
	out := make([]jev.Scored, 0, len(options))
	for i := range options {
		s := 0.0
		if i < len(src) {
			s = src[i]
		}
		out = append(out, jev.Scored{Index: i, Score: s})
	}
	return out, nil
}

func (f fake) Predict(_ context.Context, pairs []jev.Pair) ([]jev.Prediction, error) {
	if f.err != nil {
		return nil, f.err
	}
	return []jev.Prediction{{Index: 0, Scores: map[string]float64{"entailment": f.risky}}}, nil
}

func c(b Backend) Classifier { return Classifier{Backend: b, Budget: time.Second} }

func TestTierIsTheArgmaxAndConfidenceIsItsNormalisedShare(t *testing.T) {
	// Normalisation is what makes the number a confidence: three options at 0.9
	// apiece is a three-way tie, and only the normalised value can say so.
	cases := []struct {
		name     string
		scores   []float64 // fast, balanced, deep
		wantTier policy.Tier
		wantConf float64
	}{
		{"a clear deep", []float64{0.02, 0.08, 0.90}, policy.TierDeep, 0.90},
		{"a clear fast", []float64{0.90, 0.05, 0.05}, policy.TierFast, 0.90},
		{"a three-way tie is not confident", []float64{0.9, 0.9, 0.9}, policy.TierFast, 1.0 / 3.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d, _ := c(fake{tier: tc.scores, effort: []float64{1, 0, 0, 0}}).Classify(context.Background(), "task")
			if d == nil {
				t.Fatal("expected a decision")
			}
			if d.Tier != tc.wantTier {
				t.Errorf("tier = %s, want %s", d.Tier, tc.wantTier)
			}
			if d.Confidence == nil || abs(*d.Confidence-tc.wantConf) > 0.001 {
				t.Errorf("confidence = %v, want %.3f", d.Confidence, tc.wantConf)
			}
		})
	}
}

func TestEffortIsTheExpectedRungNotTheArgmax(t *testing.T) {
	// The rubric is ORDERED, so a task split evenly between "some" (1) and "a lot"
	// (2) genuinely wants 1.5 — rounding it to whichever won by 0.01 throws away the
	// ordering the rubric is made of.
	d, _ := c(fake{tier: []float64{0, 0, 1}, effort: []float64{0, 0.5, 0.5, 0}}).
		Classify(context.Background(), "task")
	if d.Effort == nil || abs(*d.Effort-1.5) > 0.001 {
		t.Fatalf("effort = %v, want 1.5", d.Effort)
	}
}

func TestRiskyIsTheRawEntailmentProbability(t *testing.T) {
	// No normalisation: the question is "how likely is this true", not "which of
	// these", so there is no set to normalise over.
	d, _ := c(fake{tier: []float64{1, 0, 0}, effort: []float64{1, 0, 0, 0}, risky: 0.93}).
		Classify(context.Background(), "drop the prod table")
	if d.Risky == nil || abs(*d.Risky-0.93) > 0.001 {
		t.Fatalf("risky = %v, want 0.93", d.Risky)
	}
}

func TestNoTierMeansNoDecisionAtAll(t *testing.T) {
	// Half a decision applied confidently is the failure this guards against: effort
	// and risk alone cannot carry one.
	d, _ := c(fake{tierErr: errors.New("boom")}).Classify(context.Background(), "task")
	if d != nil {
		t.Fatalf("expected no decision, got %+v", d)
	}
}

func TestEffortAndRiskDegradeRatherThanDiscardTheDecision(t *testing.T) {
	// All-zero effort scores give no signal, but a tier alone is still usable.
	d, _ := c(fake{tier: []float64{0, 0, 1}, effort: []float64{0, 0, 0, 0}}).
		Classify(context.Background(), "task")
	if d == nil {
		t.Fatal("a tier with no effort is still a usable decision")
	}
	if d.Effort != nil {
		t.Errorf("effort = %v, want nil rather than a fabricated 0", d.Effort)
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
