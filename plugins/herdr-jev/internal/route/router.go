package route

import (
	"context"

	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/herdrapi"
	"github.com/muthuishere/herdr-jev/internal/jev"
)

// Router is the SECONDARY feature: given a message, which open pane was it for.
//
// It is the same machinery as the headline short circuit asking a different question —
// the message is the question and each pane's distilled state is an option — which is
// why it costs almost nothing to keep. It routes a message to an agent that is ALREADY
// RUNNING, where dispatch starts a new one; those are different jobs and neither
// replaces the other.
type Router struct {
	Herdr    *herdrapi.Client
	Reranker jev.Reranker
	Cfg      config.Config
	MaxField int
}

// Snapshot gathers every pane and distils the eligible ones, reporting the rest with
// the reason they were skipped.
func (r Router) Snapshot(ctx context.Context) ([]Candidate, []Skipped, error) {
	panes, err := r.Herdr.Panes(ctx)
	if err != nil {
		return nil, nil, err
	}
	self := herdrapi.SelfPane()
	maxField := r.MaxField
	if maxField <= 0 {
		maxField = 32768
	}

	var cands []Candidate
	var skipped []Skipped
	for _, p := range panes {
		ok, why := Eligible(p, self, r.Cfg.Routing)
		if !ok {
			skipped = append(skipped, Skipped{PaneID: p.PaneID, Agent: p.Agent, Reason: why})
			continue
		}
		// A pane we cannot read is still rankable on its identity line. Degrading
		// to less signal beats refusing to route at all.
		out, _ := r.Herdr.Read(ctx, p.PaneID, r.Cfg.Routing.OutputLines)
		cands = append(cands, Distil(p, out, r.Cfg.Routing, maxField))
	}
	return cands, skipped, nil
}

// Rank scores the candidates against the message and applies the floor and the margin.
func (r Router) Rank(ctx context.Context, message string, cands []Candidate) (Decision, error) {
	if len(cands) == 0 {
		return Decide(nil, nil, r.Cfg.Routing.Floor, r.Cfg.Routing.Margin), nil
	}
	options := make([]string, len(cands))
	for i, c := range cands {
		options[i] = c.Option
	}
	scored, err := r.Reranker.Rerank(ctx, message, options)
	if err != nil {
		return Decision{}, err
	}
	scores := make(map[int]float64, len(scored))
	for _, s := range scored {
		scores[s.Index] = s.Score
	}
	return Decide(cands, scores, r.Cfg.Routing.Floor, r.Cfg.Routing.Margin), nil
}
