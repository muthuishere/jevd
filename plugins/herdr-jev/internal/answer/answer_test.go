package answer

import (
	"context"
	"errors"
	"testing"

	"github.com/muthuishere/herdr-jev/internal/jev"
)

// fake is the whole reason Backend is an interface: every test here runs with no model
// on disk, no server, and no network.
type fake struct {
	rerank  []float64
	predict []float64
	grade   jev.Grade
	err     error
}

func (f fake) Rerank(_ context.Context, _ string, options []string) ([]jev.Scored, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]jev.Scored, 0, len(options))
	for i := range options {
		s := 0.0
		if i < len(f.rerank) {
			s = f.rerank[i]
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
		s := 0.0
		if i < len(f.predict) {
			s = f.predict[i]
		}
		out = append(out, jev.Prediction{Index: i, Scores: map[string]float64{"entailment": s}})
	}
	return out, nil
}

func (f fake) Grade(context.Context, string, string, float64) (jev.Grade, error) {
	if f.err != nil {
		return jev.Grade{}, f.err
	}
	return f.grade, nil
}

func TestTryAnswersAConfidentBooleanAndDeclinesACoinFlip(t *testing.T) {
	cases := []struct {
		name    string
		predict []float64 // yes, no
		bar     float64
		verdict Verdict
		text    string
	}{
		// Asking BOTH the claim and its negation is what turns a score into a
		// confidence: a genuinely unsure model answers ~0.5/0.5 and is refused.
		{"a confident yes", []float64{0.95, 0.05}, 0.85, VerdictAnswered, "yes"},
		{"a confident no", []float64{0.04, 0.9}, 0.85, VerdictAnswered, "no"},
		{"a coin flip is refused", []float64{0.5, 0.5}, 0.85, VerdictDeclined, ""},
		{"a weak lean is refused", []float64{0.6, 0.4}, 0.85, VerdictDeclined, ""},
		{"the same weak lean passes a low bar", []float64{0.6, 0.4}, 0.55, VerdictAnswered, "yes"},
		{"both zero is not an answer", []float64{0, 0}, 0.85, VerdictUnavailable, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res := Try(context.Background(), fake{predict: c.predict},
				Request{Task: "Is the build green?", MinConfidence: c.bar})
			if res.Verdict != c.verdict {
				t.Fatalf("verdict = %s (%s), want %s", res.Verdict, res.Reason, c.verdict)
			}
			if c.text != "" && res.Answer.Text != c.text {
				t.Errorf("answer = %q, want %q", res.Answer.Text, c.text)
			}
		})
	}
}

func TestADeclinedAnswerIsStillKeptForExplanation(t *testing.T) {
	// The near misses are the evidence the bar is set right. Discarding them would
	// make the bar untunable, which is how a number nobody can justify survives.
	res := Try(context.Background(), fake{predict: []float64{0.6, 0.4}},
		Request{Task: "Is the build green?", MinConfidence: 0.85})
	if res.Verdict != VerdictDeclined {
		t.Fatalf("verdict = %s", res.Verdict)
	}
	if res.Answer == nil {
		t.Fatal("a declined answer must still carry its ranking")
	}
	if len(res.Answer.Options) != 2 {
		t.Errorf("options = %v, want yes and no with their probabilities", res.Answer.Options)
	}
}

func TestPickOneNormalisesSoATieIsNotConfident(t *testing.T) {
	// Two options at 0.9 each are NOT a confident answer, they are a tie. Reading the
	// raw score would call that 0.90 and answer it.
	tie := Try(context.Background(), fake{rerank: []float64{0.9, 0.9}},
		Request{Task: "pick", Options: []string{"a", "b"}, MinConfidence: 0.85})
	if tie.Verdict != VerdictDeclined {
		t.Errorf("a 0.9/0.9 tie must be refused, got %s at %.2f", tie.Verdict, tie.Answer.Confidence)
	}

	clear := Try(context.Background(), fake{rerank: []float64{0.95, 0.02, 0.01}},
		Request{Task: "pick", Options: []string{"a", "b", "c"}, MinConfidence: 0.85})
	if clear.Verdict != VerdictAnswered || clear.Answer.Text != "a" {
		t.Errorf("verdict = %s, answer = %+v", clear.Verdict, clear.Answer)
	}
}

func TestShapeIsCheckedBeforeConfidence(t *testing.T) {
	// The model is as confident as it can be; it is never asked, because the task is
	// generative. No confidence rescues the wrong question.
	res := Try(context.Background(), fake{predict: []float64{1, 0}, rerank: []float64{1, 0}},
		Request{Task: "Rewrite the parser", MinConfidence: 0.85})
	if res.Verdict != VerdictDeclined {
		t.Fatalf("verdict = %s", res.Verdict)
	}
	if res.Answer != nil {
		t.Error("a non-decision-shaped task must never reach the model at all")
	}
}

func TestABrokenBackendNeverBlocks(t *testing.T) {
	res := Try(context.Background(), fake{err: errors.New("connection refused")},
		Request{Task: "Is the build green?", MinConfidence: 0.85})
	if res.Verdict != VerdictUnavailable {
		t.Fatalf("verdict = %s, want unavailable so the caller dispatches normally", res.Verdict)
	}
}

func TestGradeIsOnlyEverExplicit(t *testing.T) {
	res := Try(context.Background(),
		fake{grade: jev.Grade{Pass: true, Scores: map[string]float64{"entailment": 0.93}}},
		Request{Task: "the floor is 0.55", GradeReference: "the floor defaults to 0.55", MinConfidence: 0.85})
	if res.Verdict != VerdictAnswered || res.Answer.Text != "pass" {
		t.Fatalf("verdict = %s answer = %+v", res.Verdict, res.Answer)
	}
	if res.Detection.Shape != ShapeGrade {
		t.Errorf("shape = %s", res.Detection.Shape)
	}
}
