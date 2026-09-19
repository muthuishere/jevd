package answer

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/muthuishere/herdr-jev/internal/jev"
)

// Backend is the seam. One interface, so every test here runs with no model on disk.
type Backend interface {
	Rerank(ctx context.Context, question string, options []string) ([]jev.Scored, error)
	Predict(ctx context.Context, pairs []jev.Pair) ([]jev.Prediction, error)
	Grade(ctx context.Context, answer, reference string, threshold float64) (jev.Grade, error)
}

// Option is one candidate answer with its probability, kept for provenance.
type Option struct {
	Text        string  `json:"text"`
	Probability float64 `json:"probability"`
}

// Answer is a locally-answered task AND its full provenance.
//
// A short-circuited answer that cannot be explained afterwards is not acceptable: it
// is the one path where no agent transcript exists to read back, so this struct is the
// only record that the question was ever asked. It is journalled in full.
type Answer struct {
	Shape      Shape         `json:"shape"`
	ShapeWhy   string        `json:"shape_reason"`
	Question   string        `json:"question"`
	Options    []Option      `json:"options"`
	Text       string        `json:"answer"`
	Confidence float64       `json:"confidence"`
	Latency    time.Duration `json:"-"`
	LatencyMS  int64         `json:"latency_ms"`
}

// Verdict is the three-way outcome the whole plugin turns on.
type Verdict string

const (
	// VerdictAnswered: answered locally, no agent invoked.
	VerdictAnswered Verdict = "answered"
	// VerdictDeclined: shape or confidence said no. Fall through to routing.
	VerdictDeclined Verdict = "declined"
	// VerdictUnavailable: the backend failed. Fall through, never block.
	VerdictUnavailable Verdict = "unavailable"
)

// Result is what Try returns: the verdict, the reason, and the answer when there is one.
type Result struct {
	Verdict   Verdict    `json:"verdict"`
	Reason    string     `json:"reason"`
	Detection Detection  `json:"detection"`
	Answer    *Answer    `json:"answer,omitempty"`
	// Shadow is set when the short circuit was in shadow mode: it says what it WOULD
	// have done while the task is dispatched to an agent anyway. That is how the bar
	// earns trust before it is allowed to act.
	Shadow bool `json:"shadow,omitempty"`
}

// Request is a task offered to the short circuit.
type Request struct {
	Task            string
	Options         []string
	GradeReference  string
	GradeThreshold  float64
	// MinConfidence is the answer bar: SEPARATE from and HIGHER than the routing
	// bars. Those two decide how to spend money; this one decides whether a human
	// gets a machine's answer instead of an agent's work. Different question,
	// different number.
	MinConfidence float64
	Shadow        bool
}

// Try is the short circuit. It answers the task locally, or declines and says why.
//
// It NEVER blocks and never errors upward: every failure — a dead backend, a timeout,
// a malformed body — returns VerdictUnavailable, and the caller dispatches to an agent
// exactly as it would have. Degrade, never obstruct.
func Try(ctx context.Context, b Backend, req Request) Result {
	det := Detect(req.Task, req.Options, req.GradeReference)
	res := Result{Detection: det, Shadow: req.Shadow}

	// Shape first, confidence second. No confidence rescues the wrong question.
	if det.Shape == ShapeNone {
		res.Verdict = VerdictDeclined
		res.Reason = "not decision-shaped: " + det.Reason
		return res
	}

	start := time.Now()
	ans, err := run(ctx, b, req, det)
	if err != nil {
		res.Verdict = VerdictUnavailable
		res.Reason = "the decision model could not answer (" + err.Error() + "); the agent gets the task unchanged"
		return res
	}
	ans.Latency = time.Since(start)
	ans.LatencyMS = ans.Latency.Milliseconds()
	ans.Shape, ans.ShapeWhy = det.Shape, det.Reason

	if ans.Confidence < req.MinConfidence {
		res.Verdict = VerdictDeclined
		res.Answer = ans // kept: `why` must be able to show the near miss
		res.Reason = fmt.Sprintf("decision-shaped, but confidence %.2f is below the answer bar %.2f",
			ans.Confidence, req.MinConfidence)
		return res
	}

	res.Verdict = VerdictAnswered
	res.Answer = ans
	res.Reason = fmt.Sprintf("%s, confidence %.2f clears the answer bar %.2f",
		det.Shape, ans.Confidence, req.MinConfidence)
	return res
}

func run(ctx context.Context, b Backend, req Request, det Detection) (*Answer, error) {
	switch det.Shape {
	case ShapeBoolean:
		return boolean(ctx, b, req.Task)
	case ShapePickOne:
		return pickOne(ctx, b, req.Task, det.Options)
	case ShapeGrade:
		return grade(ctx, b, req)
	}
	return nil, fmt.Errorf("unanswerable shape %q", det.Shape)
}

// boolean asks the claim and its negation as two hypotheses over the same premise.
//
// Asking BOTH, rather than reading P(entailment) alone, is what makes the number a
// confidence instead of a score. A model that is simply unsure answers ~0.5/0.5 and the
// bar refuses it; a model reading P(yes)=0.55 in isolation looks like a weak yes and
// would be indistinguishable from a coin flip.
func boolean(ctx context.Context, b Backend, task string) (*Answer, error) {
	claim := statement(task)
	pairs := []jev.Pair{
		{Premise: task, Hypothesis: claim, ID: "yes"},
		{Premise: task, Hypothesis: "It is not the case that " + lowerFirst(claim), ID: "no"},
	}
	preds, err := b.Predict(ctx, pairs)
	if err != nil {
		return nil, err
	}
	if len(preds) < 2 {
		return nil, fmt.Errorf("expected 2 predictions, got %d", len(preds))
	}
	yes, no := preds[0].Entailment(), preds[1].Entailment()
	total := yes + no
	if total <= 0 {
		return nil, fmt.Errorf("both hypotheses scored zero; nothing to choose between")
	}
	pYes, pNo := yes/total, no/total

	text, conf := "yes", pYes
	if pNo > pYes {
		text, conf = "no", pNo
	}
	return &Answer{
		Question:   task,
		Options:    []Option{{"yes", pYes}, {"no", pNo}},
		Text:       text,
		Confidence: conf,
	}, nil
}

// pickOne reranks the options and normalises the scores into a distribution.
//
// Normalisation is the point: the raw scores are independent entailment probabilities,
// so two options at 0.9 each are NOT a confident answer — they are a tie. Dividing by
// the sum turns "how well does each fit" into "how much better is the best", which is
// the question the bar is asking.
func pickOne(ctx context.Context, b Backend, task string, options []string) (*Answer, error) {
	scored, err := b.Rerank(ctx, task, options)
	if err != nil {
		return nil, err
	}
	if len(scored) == 0 {
		return nil, fmt.Errorf("no options were scored")
	}
	total := 0.0
	for _, s := range scored {
		if s.Score > 0 {
			total += s.Score
		}
	}
	if total <= 0 {
		return nil, fmt.Errorf("every option scored zero; nothing to choose between")
	}
	out := make([]Option, 0, len(scored))
	best, bestP := "", 0.0
	for _, s := range scored {
		if s.Index < 0 || s.Index >= len(options) {
			continue
		}
		p := 0.0
		if s.Score > 0 {
			p = s.Score / total
		}
		out = append(out, Option{options[s.Index], p})
		if p > bestP {
			best, bestP = options[s.Index], p
		}
	}
	return &Answer{Question: task, Options: out, Text: best, Confidence: bestP}, nil
}

// grade is the only shape that is never inferred: a reference must be handed in.
func grade(ctx context.Context, b Backend, req Request) (*Answer, error) {
	th := req.GradeThreshold
	if th <= 0 {
		th = 0.5
	}
	g, err := b.Grade(ctx, req.Task, req.GradeReference, th)
	if err != nil {
		return nil, err
	}
	ent := g.Scores["entailment"]
	text := "fail"
	conf := 1 - ent
	if g.Pass {
		text, conf = "pass", ent
	}
	return &Answer{
		Question:   req.Task,
		Options:    []Option{{"pass", ent}, {"fail", 1 - ent}},
		Text:       text,
		Confidence: conf,
	}, nil
}

// statement turns a yes/no question into the claim it asks about, so it can be a
// hypothesis. Crude on purpose: a poor rewrite scores near 0.5 and the bar refuses it,
// which is the correct outcome for a question we did not understand.
func statement(q string) string {
	s := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(q), "?"))
	return s + "."
}

func lowerFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToLower(s[:1]) + s[1:]
}

// Describe is the transcript line for a short-circuit attempt, in the same two-line
// register as the routing lines: what the model said, then what was done about it.
func (r Result) Describe() string {
	switch r.Verdict {
	case VerdictAnswered:
		if r.Shadow {
			return fmt.Sprintf("shadow: WOULD have answered locally (%.2f) — dispatching to the agent anyway", r.Answer.Confidence)
		}
		return fmt.Sprintf("answered locally in %dms (%s, confidence %.2f) — no agent invoked",
			r.Answer.LatencyMS, r.Detection.Shape, r.Answer.Confidence)
	case VerdictUnavailable:
		return "short circuit unavailable: " + r.Reason
	default:
		return "short circuit declined: " + r.Reason
	}
}
