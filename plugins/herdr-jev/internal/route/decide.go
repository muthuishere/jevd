package route

import (
	"fmt"
	"sort"
)

// Outcome is what the router decided to do. It is a closed set, journalled verbatim,
// so a month of decisions can be counted by outcome without parsing prose.
type Outcome string

const (
	OutcomeDelivered    Outcome = "delivered"
	OutcomeHeld         Outcome = "held"
	OutcomeNoCandidates Outcome = "no_candidates"
	OutcomeForced       Outcome = "forced"
	OutcomeDirected     Outcome = "directed" // --to, no ranking happened
)

// Ranked is one candidate with its score and final position.
type Ranked struct {
	Rank      int     `json:"rank"`
	Score     float64 `json:"score"`
	Candidate `json:",inline"`
}

// Decision is the whole answer: what we did, why, to whom, and what everything scored.
type Decision struct {
	Outcome Outcome  `json:"outcome"`
	Reason  string   `json:"reason"`
	Ranked  []Ranked `json:"candidates"`
	Floor   float64  `json:"floor"`
	Margin  float64  `json:"margin"`
}

// Target is the winner, or nil when nothing was chosen.
func (d Decision) Target() *Ranked {
	if len(d.Ranked) == 0 {
		return nil
	}
	switch d.Outcome {
	case OutcomeDelivered, OutcomeForced, OutcomeDirected:
		return &d.Ranked[0]
	}
	return nil
}

// Decide applies the confidence floor and the margin rule.
//
// This function is the entire point of the plugin, and it is deliberately small and
// pure so it can be tested exhaustively without a model, a pane or a socket.
//
// The failure it is organised against is ROUTING CONFIDENTLY TO THE WRONG AGENT. A
// misroute interrupts an agent mid-task, gets acted on, and produces work nobody asked
// for in a worktree nobody was watching — a revert. A hold costs one keystroke. The
// asymmetry is why both rules below prefer asking.
//
//   - FLOOR: the top score must clear it. Below the floor the model is telling us the
//     message does not really entail any pane's state, and the highest of several bad
//     matches is still a bad match.
//   - MARGIN: the top must beat the second by it. A floor alone is not enough — 0.86
//     against 0.85 clears any sane floor and is still a coin flip wearing a number.
//     Near-ties are where the misroutes live.
func Decide(cands []Candidate, scores map[int]float64, floor, margin float64) Decision {
	d := Decision{Floor: floor, Margin: margin}

	if len(cands) == 0 {
		d.Outcome = OutcomeNoCandidates
		d.Reason = "no rankable pane: nothing here runs an agent we can prompt"
		return d
	}

	for i, c := range cands {
		d.Ranked = append(d.Ranked, Ranked{Score: scores[i], Candidate: c})
	}
	// Sort by score, then by pane id so equal scores order deterministically. A
	// ranking that reshuffles between identical runs is a ranking nobody can debug.
	sort.SliceStable(d.Ranked, func(i, j int) bool {
		if d.Ranked[i].Score != d.Ranked[j].Score {
			return d.Ranked[i].Score > d.Ranked[j].Score
		}
		return d.Ranked[i].PaneID < d.Ranked[j].PaneID
	})
	for i := range d.Ranked {
		d.Ranked[i].Rank = i
	}

	top := d.Ranked[0]
	if top.Score < floor {
		d.Outcome = OutcomeHeld
		d.Reason = fmt.Sprintf("top score %.3f is below the floor %.3f — no pane matches well enough to guess", top.Score, floor)
		return d
	}
	if len(d.Ranked) > 1 {
		gap := top.Score - d.Ranked[1].Score
		if gap < margin {
			d.Outcome = OutcomeHeld
			d.Reason = fmt.Sprintf("top %.3f beats second %.3f by only %.3f, under the margin %.3f — too close to call",
				top.Score, d.Ranked[1].Score, gap, margin)
			return d
		}
		d.Outcome = OutcomeDelivered
		d.Reason = fmt.Sprintf("top %.3f clears the floor %.3f and beats second %.3f by %.3f",
			top.Score, floor, d.Ranked[1].Score, gap)
		return d
	}

	d.Outcome = OutcomeDelivered
	d.Reason = fmt.Sprintf("top %.3f clears the floor %.3f; it is the only candidate", top.Score, floor)
	return d
}

// Force rewrites a decision as a deliberate human override.
//
// It exists for the human who already knows which pane they meant. It is never a
// default and never a config key: a floor you can switch off in a file is a floor that
// gets switched off once, during a demo, and never switched back.
func Force(d Decision) Decision {
	if len(d.Ranked) == 0 {
		return d
	}
	d.Outcome = OutcomeForced
	d.Reason = fmt.Sprintf("--force: delivered to the top candidate (%.3f) regardless of floor %.3f and margin %.3f",
		d.Ranked[0].Score, d.Floor, d.Margin)
	return d
}
