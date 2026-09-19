package dispatch

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/muthuishere/herdr-jev/internal/agentkind"
	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/jev"
	"github.com/muthuishere/herdr-jev/internal/policy"
)

// registryEntailLabel is deliberately NOT "entailment". The label set is registry
// config, so a fake that used the obvious name would be more permissive than the real
// server — which is how a wrong-answer bug survives a green suite.
const registryEntailLabel = "entails"

// fake serves the tier question (3 options), the effort rubric (4 options), any
// pick-one, and the boolean/risky predicts — enough to drive all three stages with no
// model on disk.
type fake struct {
	tier    []float64
	effort  []float64
	boolean []float64
	risky   float64
	err     error
}

func (f fake) Rerank(_ context.Context, _ string, options []string) ([]jev.Scored, error) {
	if f.err != nil {
		return nil, f.err
	}
	src := f.boolean
	switch len(options) {
	case 3:
		src = f.tier
	case 4:
		src = f.effort
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
	out := make([]jev.Prediction, 0, len(pairs))
	for i := range pairs {
		s := f.risky
		if len(pairs) == 2 { // a boolean short circuit: yes then no
			s = 0
			if i < len(f.boolean) {
				s = f.boolean[i]
			}
		}
		out = append(out, jev.Prediction{Index: i, EntailmentLabel: registryEntailLabel,
			Scores: map[string]float64{registryEntailLabel: s, "contra": 1 - s}})
	}
	return out, nil
}

func (f fake) Grade(context.Context, string, string, float64) (jev.Grade, error) {
	return jev.Grade{}, errors.New("not used")
}

func base() config.Config {
	c := config.Default()
	c.Agents = map[string]agentkind.Kind{"claude": agentkind.Defaults()["claude"]}
	return c
}

func TestTheThreeStagesHappenInOrder(t *testing.T) {
	deepTier := []float64{0.01, 0.04, 0.95}

	cases := []struct {
		name      string
		cfg       func(config.Config) config.Config
		backend   Backend
		req       Request
		wantStage Stage
		wantLine  string
	}{
		{
			name: "1. a confident decision-shaped question is answered locally, no agent invoked",
			cfg: func(c config.Config) config.Config {
				c.Answer.Enabled = true
				return c
			},
			backend:   fake{boolean: []float64{0.97, 0.02}, tier: deepTier},
			req:       Request{Task: "Is the build green?", Kind: "claude"},
			wantStage: StageAnswered,
			wantLine:  "no agent invoked",
		},
		{
			name: "1b. the short circuit is OFF by default, so the same question routes",
			// Answering instead of invoking the user's agent is a large claim to
			// make on their behalf; installing the plugin must not make it.
			backend:   fake{boolean: []float64{0.97, 0.02}, tier: deepTier, effort: []float64{0, 0, 0, 1}},
			req:       Request{Task: "Is the build green?", Kind: "claude"},
			wantStage: StageRouted,
		},
		{
			name: "1c. shadow mode says what it WOULD have done and dispatches anyway",
			cfg: func(c config.Config) config.Config {
				c.Answer.Enabled, c.Answer.Shadow = true, true
				return c
			},
			backend:   fake{boolean: []float64{0.97, 0.02}, tier: deepTier, effort: []float64{0, 0, 0, 1}},
			req:       Request{Task: "Is the build green?", Kind: "claude"},
			wantStage: StageRouted,
			wantLine:  "WOULD have answered locally",
		},
		{
			name: "2. a generative task is never answered; it is routed",
			cfg: func(c config.Config) config.Config {
				c.Answer.Enabled = true
				return c
			},
			backend:   fake{boolean: []float64{0.99, 0.0}, tier: deepTier, effort: []float64{0, 0, 0, 1}},
			req:       Request{Task: "Rewrite the parser to stream", Kind: "claude"},
			wantStage: StageRouted,
		},
		{
			name:      "3. a dead backend passes the task through untouched",
			backend:   fake{err: errors.New("connection refused")},
			req:       Request{Task: "Rewrite the parser", Kind: "claude"},
			wantStage: StagePassed,
		},
		{
			name:      "3b. no backend at all passes through",
			backend:   nil,
			req:       Request{Task: "Rewrite the parser", Kind: "claude"},
			wantStage: StagePassed,
		},
		{
			name:      "3c. an unknown agent kind passes through rather than guessing a model",
			backend:   fake{tier: deepTier, effort: []float64{0, 0, 0, 1}},
			req:       Request{Task: "Rewrite the parser", Kind: "nosuchagent"},
			wantStage: StagePassed,
		},
		{
			name: "3d. a decision that changes nothing passes through and says what it wanted",
			// The session is already on the deep tier's model at xhigh.
			backend: fake{tier: deepTier, effort: []float64{0, 0, 0, 1}},
			req: Request{Task: "Rewrite the parser", Kind: "claude",
				Current: policy.Current{Model: "opus", Effort: "xhigh"}},
			wantStage: StagePassed,
			wantLine:  "kept opus/xhigh",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := base()
			if c.cfg != nil {
				cfg = c.cfg(cfg)
			}
			var b Backend
			if c.backend != nil {
				b = c.backend
			}
			p := Dispatcher{Cfg: cfg, Backend: b}.Decide(context.Background(), c.req)
			if p.Stage != c.wantStage {
				t.Fatalf("stage = %s (%s)\nlines: %s", p.Stage, p.Reason, strings.Join(p.Lines, " | "))
			}
			if c.wantLine != "" && !strings.Contains(strings.Join(p.Lines, " | "), c.wantLine) {
				t.Errorf("no line mentioning %q; got: %s", c.wantLine, strings.Join(p.Lines, " | "))
			}
		})
	}
}

func TestRoutingProducesRealAgentArguments(t *testing.T) {
	// The whole model-routing feature is worthless if the plan does not turn into
	// arguments a process actually takes.
	b := fake{tier: []float64{0.01, 0.04, 0.95}, effort: []float64{0, 0, 0, 1}}
	p := Dispatcher{Cfg: base(), Backend: b}.Decide(context.Background(),
		Request{Task: "Design the sharding scheme", Kind: "claude", Current: policy.Current{Model: "haiku"}})
	if p.Stage != StageRouted {
		t.Fatalf("stage = %s (%s)", p.Stage, p.Reason)
	}
	if args := strings.Join(p.Launch.Args, " "); args != "--model opus" {
		t.Errorf("args = %q, want %q", args, "--model opus")
	}
}

func TestTheSwitchesStopTheActionButNotTheLog(t *testing.T) {
	// Applying the switches AFTER the policy ran is what lets the log still say what
	// it wanted when it is not allowed to want it.
	cfg := base()
	cfg.Policy.RouteModel = false
	cfg.Policy.RouteEffort = false
	b := fake{tier: []float64{0.01, 0.04, 0.95}, effort: []float64{0, 0, 0, 1}}
	p := Dispatcher{Cfg: cfg, Backend: b}.Decide(context.Background(),
		Request{Task: "Design the sharding scheme", Kind: "claude", Current: policy.Current{Model: "haiku"}})
	if p.Stage != StagePassed {
		t.Fatalf("stage = %s", p.Stage)
	}
	if p.Decision == nil || p.Decision.Tier != policy.TierDeep {
		t.Error("the classification must still be recorded even when nothing may be applied")
	}
}

func TestSavingsCountsWhatTheShortCircuitBought(t *testing.T) {
	// This number IS the product: without it, "it sometimes answers things itself"
	// is a claim rather than a result.
	records := []Record{
		{Plan: Plan{Stage: StageAnswered}},
		{Plan: Plan{Stage: StageAnswered}},
		{Plan: Plan{Stage: StageRouted}},
		{Plan: Plan{Stage: StagePassed}},
	}
	s := Count(records)
	if s.Total != 4 || s.Answered != 2 || s.Routed != 1 || s.Passed != 1 {
		t.Fatalf("savings = %+v", s)
	}
}
