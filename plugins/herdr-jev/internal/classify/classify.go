// Package classify asks the decision model the three questions the policy needs, and
// turns its answers into a policy.Decision.
//
// The reference implementation this policy comes from posts one request with three
// typed questions to a hosted API, shipping the prompt text off the machine. We ask the
// same three questions of a cross-encoder running on loopback, as NLI hypotheses — and
// NOTHING LEAVES THE MACHINE. That is not a side effect of the port; it is the reason
// for it. A router that reads every task you type is a router you have to trust with
// every task you type.
//
// The mapping from entailment to the confidence the policy reads is the only real
// design work here, and it is stated in full in SPEC.md §5.
package classify

import (
	"context"
	"time"

	"github.com/muthuishere/herdr-jev/internal/jev"
	"github.com/muthuishere/herdr-jev/internal/policy"
)

// Backend is the seam: one interface, so the whole classifier is testable with no model.
type Backend interface {
	Rerank(ctx context.Context, question string, options []string) ([]jev.Scored, error)
	Predict(ctx context.Context, pairs []jev.Pair) ([]jev.Prediction, error)
}

// Classifier asks the three questions.
type Classifier struct {
	Backend Backend
	// Budget is the latency budget for the whole classification. Past it the task is
	// dispatched exactly as it was built: a router that makes every task wait is a
	// tax, and the saving it is chasing is not worth a visible pause.
	Budget time.Duration
}

// Classify answers tier, effort and risky, or returns nil.
//
// nil is a first-class outcome, not an error: every caller treats "no decision" as
// "leave the request alone", so a dead backend and a slow one and a confused one all
// land in the same, safe place.
func (c Classifier) Classify(ctx context.Context, task string) (*policy.Decision, time.Duration) {
	start := time.Now()
	if c.Budget > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.Budget)
		defer cancel()
	}

	tier, tierConf, err := c.tier(ctx, task)
	if err != nil {
		// Without a tier there is no decision at all. Effort and risk alone cannot
		// carry one, and half a decision applied confidently is the failure mode
		// this package exists to avoid.
		return nil, time.Since(start)
	}

	d := &policy.Decision{Tier: tier, Confidence: tierConf}

	// Effort and risk are best-effort: a decision with a tier and no effort is a
	// usable decision, so their failures degrade rather than discard.
	if score, conf, err := c.effort(ctx, task); err == nil {
		d.Effort, d.EffortConfidence = score, conf
	}
	if risky, err := c.risky(ctx, task); err == nil {
		d.Risky = risky
	}
	return d, time.Since(start)
}

// tier reranks the task against the three descriptions of the WORK.
//
// The cross-encoder never sees a model name — it is shown three descriptions of work
// and asked which the task is. Normalising the scores is what makes the winner's share
// a CONFIDENCE: three options at 0.9 apiece is not a confident answer, it is a
// three-way tie, and only the normalised value can tell them apart.
func (c Classifier) tier(ctx context.Context, task string) (policy.Tier, *float64, error) {
	options := make([]string, len(policy.TierOrder))
	for i, t := range policy.TierOrder {
		options[i] = policy.TierCriteria[t]
	}
	scored, err := c.Backend.Rerank(ctx, task, options)
	if err != nil {
		return "", nil, err
	}
	idx, conf := argmax(scored, len(options))
	if idx < 0 {
		return "", nil, errNoSignal
	}
	return policy.TierOrder[idx], conf, nil
}

// effort reranks the task against the four-level rubric and takes the EXPECTED level.
//
// An expectation, not the argmax: the rubric is ordered, so a task split evenly between
// "some" and "a lot" genuinely wants something between them, and rounding that to
// whichever won by 0.01 throws away the ordering the rubric is made of. The confidence
// reported alongside is still the winner's share, because that is what the bars read.
func (c Classifier) effort(ctx context.Context, task string) (*float64, *float64, error) {
	scored, err := c.Backend.Rerank(ctx, task, policy.EffortRubric)
	if err != nil {
		return nil, nil, err
	}
	probs, total := normalise(scored, len(policy.EffortRubric))
	if total <= 0 {
		return nil, nil, errNoSignal
	}
	expected := 0.0
	for i, p := range probs {
		expected += float64(i) * p
	}
	_, conf := argmax(scored, len(policy.EffortRubric))
	return &expected, conf, nil
}

// risky asks the one question that is not a confidence question.
//
// A single hypothesis over the task as premise: P(entailment) IS the probability, which
// is exactly the `noul` the reference policy reads. No normalisation, because there is
// no set to normalise over — the question is "how likely is this true", not "which of
// these".
func (c Classifier) risky(ctx context.Context, task string) (*float64, error) {
	preds, err := c.Backend.Predict(ctx, []jev.Pair{
		{Premise: task, Hypothesis: policy.RiskyHypothesis, ID: "risky"},
	})
	if err != nil {
		return nil, err
	}
	if len(preds) == 0 {
		return nil, errNoSignal
	}
	p := preds[0].Entailment()
	return &p, nil
}

type noSignal struct{}

func (noSignal) Error() string { return "the model returned no usable signal" }

var errNoSignal = noSignal{}

// normalise turns independent entailment scores into a distribution over the option
// set. Negative and missing scores read as zero.
func normalise(scored []jev.Scored, n int) ([]float64, float64) {
	probs := make([]float64, n)
	total := 0.0
	for _, s := range scored {
		if s.Index < 0 || s.Index >= n || s.Score <= 0 {
			continue
		}
		probs[s.Index] = s.Score
		total += s.Score
	}
	if total <= 0 {
		return probs, 0
	}
	for i := range probs {
		probs[i] /= total
	}
	return probs, total
}

// argmax returns the winning index and its normalised share, or -1 when there is no
// signal at all. The share is the confidence every bar in the policy reads.
func argmax(scored []jev.Scored, n int) (int, *float64) {
	probs, total := normalise(scored, n)
	if total <= 0 {
		return -1, nil
	}
	best, bestP := -1, -1.0
	for i, p := range probs {
		if p > bestP {
			best, bestP = i, p
		}
	}
	conf := bestP
	return best, &conf
}
