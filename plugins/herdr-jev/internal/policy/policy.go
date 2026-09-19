// Package policy is herdr-jev's decision logic, and nothing else.
//
// No HTTP, no Herdr, no I/O of any kind: this package takes what the decision model
// answered and the state a request was built with, and says what to change. Every
// interesting rule in this plugin lives here so it can be tested exhaustively with no
// model on disk and no pane open.
//
// It is a faithful Go port of the policy in TypeSafe's `jev-model-router` Claude Code
// mod, whose asymmetric-confidence design we adopt wholesale (see docs/adr/0002). The
// differences are stated in SPEC.md §2 and are about WHERE the answer comes from and
// WHAT it is applied to, never about how the decision is made.
package policy

import (
	"fmt"
	"math"
	"strings"
)

// Tier is a judgement about the WORK, never about a model.
//
// The decision model is never shown a model name. It is asked which description of the
// task fits, and the mapping from that to an actual model id happens here, afterwards,
// from per-agent-kind config. A classifier that saw "haiku" would be ranking brand
// names; this one ranks work.
type Tier string

const (
	TierFast     Tier = "fast"
	TierBalanced Tier = "balanced"
	TierDeep     Tier = "deep"
)

// TierOrder is cheapest first. Index in this slice IS the rank.
var TierOrder = []Tier{TierFast, TierBalanced, TierDeep}

// TierCriteria is how each tier is described to the decision model. Deliberately about
// the shape of the work, not about model names.
var TierCriteria = map[Tier]string{
	TierFast:     "Mechanical and local: read or summarise a file, run one command, rename a symbol, answer something already in context.",
	TierBalanced: "Ordinary engineering: implement a well-specified change across a few files, write tests, fix a clearly described bug, review a small diff.",
	TierDeep:     "Hard or high-stakes: architecture and design, debugging a failure whose cause is unknown, security, data migrations, concurrency, anything touching production or money.",
}

// Effort is a reasoning level, cheapest first.
type Effort string

// EffortOrder is both the ladder and the set of values this router is allowed to ask
// for. "max" ranks above all of them (see EffortRank) without joining the list.
var EffortOrder = []Effort{"low", "medium", "high", "xhigh"}

// EffortRubric is the four-level scale the decision model scores against.
var EffortRubric = []string{"almost none", "some", "a lot", "as much as possible"}

// RiskyHypothesis is the risk question, and its exact wording is load-bearing.
//
// It asks about the ACT, not the subject. The obvious wording — "the task touches
// production, money, credentials" — scores near-certain on "add a refund endpoint that
// calls Stripe", which is ordinary code that merely happens to be about money, and
// would escalate it past a high-confidence answer of the balanced tier. Writing code
// that deals with dangerous things is not the same as doing a dangerous thing.
const RiskyHypothesis = "Carrying out this task would itself change production, move real money, or alter data that cannot be restored. Writing or testing code that deals with such things, without running it against the real system, does not count."

// RiskyForceThreshold is where risk stops being a confidence question.
//
// Above it the deep tier and real reasoning are taken regardless of what the cheaper
// answer said and regardless of either confidence bar. Carrying out something final is
// never worth the saving.
const RiskyForceThreshold = 0.7

// Decision is what the decision model answered, before any policy is applied.
//
// Every confidence is a pointer because "the backend reported none" is a real and
// consequential state, not a zero: a nil confidence may move a request UP but never
// DOWN, and collapsing it into 0.0 would silently forbid both.
type Decision struct {
	Tier Tier `json:"tier"`
	// Confidence in the tier, or nil when the backend reported none.
	Confidence *float64 `json:"confidence"`
	// Risky is P(true) that carrying the task out would itself be costly or final.
	Risky *float64 `json:"risky"`
	// Effort is 0..3 along the rubric, or nil when absent.
	Effort           *float64 `json:"effort"`
	EffortConfidence *float64 `json:"effort_confidence"`
}

// Tiers maps each tier to a model name for one agent kind.
//
// Per agent kind, not global: claude, codex, gemini and the rest name their models
// differently, and a hardcoded haiku/sonnet/opus is a router that works for exactly one
// agent. The decision model stays blind to all of it.
type Tiers struct {
	Fast     string `toml:"fast" json:"fast"`
	Balanced string `toml:"balanced" json:"balanced"`
	Deep     string `toml:"deep" json:"deep"`
}

// Get returns the configured model for a tier.
func (t Tiers) Get(tier Tier) string {
	switch tier {
	case TierFast:
		return t.Fast
	case TierBalanced:
		return t.Balanced
	case TierDeep:
		return t.Deep
	}
	return ""
}

// Config is the two numbers the whole policy turns on, plus the tier mapping.
type Config struct {
	Tiers Tiers
	// MinUpgradeConfidence: how sure the decision must be to spend MORE (a bigger
	// model, more reasoning). Being wrong here costs money, so the bar is low.
	MinUpgradeConfidence float64
	// MinDowngradeConfidence: how sure it must be to spend LESS. Being wrong here
	// means a real task handled by too small a model or too little thought, so the
	// bar is high.
	//
	// These two numbers being DIFFERENT is the central idea of this policy. The two
	// mistakes do not cost the same, so they do not clear the same bar.
	MinDowngradeConfidence float64
	// FamilyHints are per-agent-kind substrings used to rank a model id that matches
	// no configured tier name, e.g. {"haiku":0,"sonnet":1,"opus":2}.
	FamilyHints map[string]int
}

// Routing is the answer: either field may be empty, meaning "leave it as it is".
type Routing struct {
	Model  string
	Effort Effort
	Reason string
}

// Changed reports whether the policy is asking for anything at all.
func (r Routing) Changed() bool { return r.Model != "" || r.Effort != "" }

// EffortLevel maps a rubric score (0..3) onto the reasoning ladder.
func EffortLevel(score float64) Effort {
	i := int(math.Round(score))
	if i < 0 {
		i = 0
	}
	if i > len(EffortOrder)-1 {
		i = len(EffortOrder) - 1
	}
	return EffortOrder[i]
}

// EffortRank is where a reasoning level sits on the ladder, or nil when its place
// cannot be known.
//
// A NUMERIC effort is the caller's own scale (a token budget, say), not this ladder, so
// it has no rank here and is left alone entirely. "max" is above every rung the rubric
// can produce, so it ranks above them without joining EffortOrder — leaving "max" is
// therefore a downgrade and needs the high bar.
func EffortRank(effort any) *int {
	s, ok := effort.(string)
	if !ok {
		if e, ok2 := effort.(Effort); ok2 {
			s = string(e)
		} else {
			return nil
		}
	}
	if s == "max" {
		n := len(EffortOrder)
		return &n
	}
	for i, e := range EffortOrder {
		if string(e) == s {
			r := i
			return &r
		}
	}
	return nil
}

// RankOf is where a model id sits on the tier ladder: configured tier names first, then
// the agent kind's family hints.
//
// Nil when it matches none — and an unrecognised id is then treated as an UPGRADE
// rather than guessed at, because a change whose direction cannot be known must get the
// gentler bar, never the strict one. Guessing the direction wrong is how a downgrade
// sneaks past the bar that exists to stop it.
func RankOf(model string, tiers Tiers, hints map[string]int) *int {
	lowered := strings.ToLower(strings.TrimSpace(model))
	if lowered == "" {
		return nil
	}
	for i, tier := range TierOrder {
		configured := strings.ToLower(tiers.Get(tier))
		if configured != "" && strings.Contains(lowered, configured) {
			r := i
			return &r
		}
	}
	// Longest hint first: "gpt-5-mini" must win over "gpt-5" when both are hints.
	best, bestLen := -1, -1
	for hint, rank := range hints {
		h := strings.ToLower(hint)
		if h != "" && strings.Contains(lowered, h) && len(h) > bestLen {
			best, bestLen = rank, len(h)
		}
	}
	if best >= 0 {
		return &best
	}
	return nil
}

// allowed reports whether a change of rank passes its threshold.
//
// Both directions are allowed; they just do not have to clear the same bar. A move
// whose direction cannot be told (an unrecognised current value) is treated as an
// upgrade.
func allowed(wanted int, current *int, confidence *float64, cfg Config) bool {
	if current != nil && wanted == *current {
		return false
	}
	isDowngrade := current != nil && wanted < *current
	bar := cfg.MinUpgradeConfidence
	if isDowngrade {
		bar = cfg.MinDowngradeConfidence
	}
	// A backend that reports no confidence at all clears the upgrade bar but never
	// the downgrade one: spending less on an unmeasured hunch is the bad trade.
	if confidence == nil {
		return !isDowngrade
	}
	return *confidence >= bar
}

// Current is the state a request was built with, before the policy touches it.
//
// Effort is `any` because it may legitimately be absent (nil), a rung of our ladder (a
// string), or the caller's own numeric scale — and those three are not the same thing.
type Current struct {
	Model  string
	Effort any
}

// Route turns a decision into a model and a reasoning level, either of which may be
// empty to leave the request as it is. Both can move in either direction.
func Route(d *Decision, current Current, cfg Config) Routing {
	if d == nil {
		return Routing{Reason: "no decision"}
	}

	tier := d.Tier
	effortScore := d.Effort
	forced := false

	// Carrying out something final is never worth the saving: take the deep tier and
	// real reasoning, whatever the cheaper answer said, and skip the thresholds —
	// this is the one case that is not a confidence question.
	if d.Risky != nil && *d.Risky > RiskyForceThreshold {
		tier = TierDeep
		base := 0.0
		if effortScore != nil {
			base = *effortScore
		}
		v := math.Max(base, 2)
		effortScore = &v
		forced = true
	}

	wantedTier := tierIndex(tier)
	currentTier := RankOf(current.Model, cfg.Tiers, cfg.FamilyHints)
	wantedModel := cfg.Tiers.Get(tier)

	model := ""
	if wantedModel != "" && wantedModel != current.Model &&
		(forced || allowed(wantedTier, currentTier, d.Confidence, cfg)) {
		model = wantedModel
	}

	var effort Effort
	if effortScore != nil {
		currentRank := EffortRank(current.Effort)
		wantedRank := effortIndex(EffortLevel(*effortScore))

		// Risk RAISES the floor; it must never lower one. Forcing only skips the
		// thresholds, so without this clamp a task already at xhigh or max but
		// rated mechanically simple would be pulled down to high with no
		// confidence check at all — the exact opposite of what the rule is for.
		if forced && currentRank != nil && *currentRank > wantedRank {
			wantedRank = *currentRank
		}

		// A numeric effort is the caller's own scale, not this ladder; leave it.
		_, isNumeric := numeric(current.Effort)
		comparable := !isNumeric
		capped := min(wantedRank, len(EffortOrder)-1)
		wanted := EffortOrder[capped]

		if comparable && (currentRank == nil || *currentRank != wantedRank) &&
			(forced || allowed(wantedRank, currentRank, d.EffortConfidence, cfg)) {
			effort = wanted
		}
	}

	said := "confidence n/d"
	if d.Confidence != nil {
		said = fmt.Sprintf("confidence %.2f", *d.Confidence)
	}

	if model == "" && effort == "" {
		// Naming what it wanted and what it kept is the whole point of this line.
		// Without it, a router that classified and decided to leave the request
		// alone is indistinguishable from one that never ran.
		kept := current.Model
		if current.Effort != nil {
			kept += "/" + fmt.Sprintf("%v", current.Effort)
		}
		want := wantedModel
		if effortScore != nil {
			want += "/" + string(EffortLevel(*effortScore))
		}
		return Routing{Reason: fmt.Sprintf("kept %s, wanted %s (%s)", kept, want, said)}
	}

	if forced {
		return Routing{Model: model, Effort: effort, Reason: fmt.Sprintf("%s, forced by risk", tier)}
	}
	return Routing{Model: model, Effort: effort, Reason: fmt.Sprintf("%s (%s)", tier, said)}
}

func tierIndex(t Tier) int {
	for i, v := range TierOrder {
		if v == t {
			return i
		}
	}
	return 0
}

func effortIndex(e Effort) int {
	for i, v := range EffortOrder {
		if v == e {
			return i
		}
	}
	return 0
}

func numeric(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case float64:
		return n, true
	case float32:
		return float64(n), true
	}
	return 0, false
}

// --- transparency ------------------------------------------------------------
//
// The log lines below are a CONTRACT, not decoration. Nothing else in Herdr shows what
// a router decided — the model and effort it picks are arguments to a process, so no
// status line or header ever moves — and a router whose work is invisible is one that
// cannot be trusted or tuned.

func reported(v *float64) string {
	if v == nil {
		return "n/d"
	}
	return fmt.Sprintf("%.2f", *v)
}

// DescribeSetup is the one-time line saying the router is alive, which backend answers
// it, and what it is allowed to set.
//
// Without it, a router that loaded and a router that never loaded are told apart only
// by the absence of later lines, which is not evidence of anything.
func DescribeSetup(backend, url string, switches []string) string {
	where := backend
	if url != "" {
		where = fmt.Sprintf("%s (%s)", backend, url)
	}
	on := "nothing, every switch is off"
	if len(switches) > 0 {
		on = strings.Join(switches, ", ")
	}
	return fmt.Sprintf("ready on %s; routing %s", where, on)
}

// DescribeDecision is what the decision model answered, BEFORE any policy touches it.
//
// It is a separate line from what the policy did, on purpose: "the classification
// happened" and "the classification was acted on" are different facts, and one log line
// cannot fail to distinguish them without hiding the case where a working router
// deliberately declined to act.
func DescribeDecision(d *Decision, ms float64) string {
	took := ""
	if ms >= 0 {
		took = fmt.Sprintf(" · %dms", int(math.Round(ms)))
	}
	if d == nil {
		return "no answer" + took
	}
	parts := []string{fmt.Sprintf("tier %s (%s)", d.Tier, reported(d.Confidence))}
	if d.Effort != nil {
		parts = append(parts, fmt.Sprintf("effort %.1f → %s (%s)",
			*d.Effort, EffortLevel(*d.Effort), reported(d.EffortConfidence)))
	}
	if d.Risky != nil {
		parts = append(parts, fmt.Sprintf("risky %s", reported(d.Risky)))
	}
	return strings.Join(parts, " · ") + took
}

// DescribeStatus is the short line: the last thing the router did.
func DescribeStatus(d *Decision, r Routing) string {
	if d == nil {
		return "jev · no answer"
	}
	asked := fmt.Sprintf("%s %s", d.Tier, reported(d.Confidence))
	if !r.Changed() {
		return fmt.Sprintf("jev · %s · unchanged", asked)
	}
	var to []string
	if r.Model != "" {
		to = append(to, r.Model)
	}
	if r.Effort != "" {
		to = append(to, string(r.Effort))
	}
	return fmt.Sprintf("jev · %s → %s", asked, strings.Join(to, "/"))
}
