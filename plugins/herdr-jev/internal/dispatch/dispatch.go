// Package dispatch is the three-way decision, in order, and the only place it lives.
//
//	1. ANSWER   — decision-shaped and confident? openjev answers it. No agent runs.
//	2. ROUTE    — otherwise pick the tier and effort, and start the agent with them.
//	3. PASS     — any failure, timeout or ambiguity: the agent gets the task exactly
//	              as it was built.
//
// Stage 3 is not an error path bolted on the side. It is the default that stages 1 and
// 2 have to earn their way out of: every branch below that cannot prove what it wants
// falls back to it, so a broken openjev, a slow one, a confused one and an absent one
// are all indistinguishable from not having installed this plugin at all.
package dispatch

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/muthuishere/herdr-jev/internal/agentkind"
	"github.com/muthuishere/herdr-jev/internal/answer"
	"github.com/muthuishere/herdr-jev/internal/classify"
	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/policy"
)

// Stage is which of the three happened.
type Stage string

const (
	StageAnswered Stage = "answered_locally"
	StageRouted   Stage = "routed"
	StagePassed   Stage = "passed_through"
)

// Plan is the decided outcome of a dispatch, before anything is executed.
//
// Deciding and executing are separate so `--dry-run` is the SAME code path as a real
// dispatch minus one call, rather than a parallel implementation that can drift from
// the one that actually runs.
type Plan struct {
	Stage  Stage  `json:"stage"`
	Reason string `json:"reason"`

	// Answer is set only when Stage is StageAnswered.
	Answer *answer.Answer `json:"answer,omitempty"`
	// ShortCircuit is always set when the short circuit ran, including when it
	// declined or was in shadow mode — the near misses are the evidence the bar is
	// set right, and discarding them would make the bar untunable.
	ShortCircuit *answer.Result `json:"short_circuit,omitempty"`

	// Decision is what the model said, before policy. Nil when nothing was asked.
	Decision *policy.Decision `json:"decision,omitempty"`
	// Routing is what the policy then did about it.
	Routing policy.Routing `json:"routing"`
	// Launch is the concrete args and env the agent process would start with.
	Launch agentkind.Launch `json:"launch"`

	Kind          string        `json:"kind"`
	ClassifyTime  time.Duration `json:"-"`
	ClassifyMS    int64         `json:"classify_ms"`
	// Lines is the transcript record: what the model said, then what was done. Two
	// lines, never one, because "it classified" and "it acted" are different facts
	// and a single line cannot fail to conflate them.
	Lines []string `json:"lines"`
}

// Request is one task to dispatch.
type Request struct {
	Task string
	// Kind is the agent kind that would run it, e.g. "claude". It decides the tier
	// mapping and the flags, and is required — a router that does not know which
	// agent it is arguing about cannot pick its model.
	Kind string
	// Current is the model and effort the agent would otherwise start with. Empty
	// model is normal (the agent's own default) and simply has no knowable rank.
	Current policy.Current
	// Options / GradeReference feed the short circuit's shape detection only.
	Options        []string
	GradeReference string
}

// Backend is everything the dispatcher needs from openjev. One interface, so the whole
// three-way decision is testable with no model on disk.
type Backend interface {
	answer.Backend
	classify.Backend
}

// Dispatcher decides. It never touches Herdr; executing the plan is the caller's job,
// which keeps the decision pure and the side effects in one place.
type Dispatcher struct {
	Cfg     config.Config
	Backend Backend
}

// Decide runs the three stages in order and returns the plan.
//
// It returns no error, ever. There is no failure of this function that is not simply
// "pass the task through", and returning an error would invite a caller to abort a
// task the user asked for because a classifier was unwell.
func (d Dispatcher) Decide(ctx context.Context, req Request) Plan {
	p := Plan{Kind: req.Kind, Stage: StagePassed, Reason: "nothing decided"}

	// ---- stage 1: answer locally -------------------------------------------
	if d.Cfg.Answer.Enabled || d.Cfg.Answer.Shadow {
		if d.Backend == nil {
			p.Lines = append(p.Lines, "short circuit skipped: no openjev backend")
		} else {
			res := answer.Try(ctx, d.Backend, answer.Request{
				Task:           req.Task,
				Options:        req.Options,
				GradeReference: req.GradeReference,
				GradeThreshold: d.Cfg.Answer.GradeThreshold,
				MinConfidence:  d.Cfg.Answer.MinConfidence,
				// Shadow wins over Enabled: shadow exists precisely so the
				// short circuit can be watched without acting, and a config
				// with both set must watch, not act.
				Shadow: d.Cfg.Answer.Shadow,
			})
			p.ShortCircuit = &res
			p.Lines = append(p.Lines, "jev: "+res.Describe())

			if res.Verdict == answer.VerdictAnswered && !d.Cfg.Answer.Shadow && d.Cfg.Answer.Enabled {
				p.Stage, p.Reason, p.Answer = StageAnswered, res.Reason, res.Answer
				p.Lines = append(p.Lines, "answered locally — no agent invoked, 0 tokens, nothing left the machine")
				return p
			}
		}
	}

	// ---- stage 2: route ----------------------------------------------------
	pcfg, kind, err := d.Cfg.PolicyFor(req.Kind)
	if err != nil {
		p.Reason = err.Error()
		p.Lines = append(p.Lines, "passed through: "+err.Error())
		return p
	}
	if d.Backend == nil {
		p.Reason = "no openjev backend; the agent gets the task unchanged"
		p.Lines = append(p.Lines, "passed through: "+p.Reason)
		return p
	}

	decision, took := classify.Classifier{Backend: d.Backend, Budget: d.Cfg.Budget()}.
		Classify(ctx, req.Task)
	p.Decision, p.ClassifyTime, p.ClassifyMS = decision, took, took.Milliseconds()
	p.Lines = append(p.Lines, "jev: "+policy.DescribeDecision(decision, float64(took.Milliseconds())))

	if decision == nil {
		p.Reason = "no classification within the budget; the agent gets the task unchanged"
		p.Lines = append(p.Lines, "passed through: "+p.Reason)
		return p
	}

	routing := policy.Route(decision, req.Current, pcfg)
	// The switches are applied AFTER the policy ran, not before, so the log still
	// says what it wanted even when it is not allowed to want it.
	if !d.Cfg.Policy.RouteModel {
		routing.Model = ""
	}
	if !d.Cfg.Policy.RouteEffort {
		routing.Effort = ""
	}
	p.Routing = routing

	if !routing.Changed() {
		p.Reason = routing.Reason
		p.Lines = append(p.Lines, "passed through: "+routing.Reason)
		return p
	}

	p.Launch = kind.Build(routing.Model, routing.Effort)
	p.Stage, p.Reason = StageRouted, routing.Reason
	p.Lines = append(p.Lines, fmt.Sprintf("routed → %s", policy.DescribeStatus(decision, routing)))
	for _, n := range p.Launch.Notes {
		p.Lines = append(p.Lines, "note: "+n)
	}
	return p
}

// --- journal ------------------------------------------------------------------

// Record is one dispatch, journalled.
//
// The saving is the product, so it has to be countable: every record carries its stage,
// which is all `status` needs to say how many tasks never reached an agent.
type Record struct {
	V     int       `json:"v"`
	TS    time.Time `json:"ts"`
	Task  string    `json:"task"`
	Plan  Plan      `json:"plan"`
	DryRun bool     `json:"dry_run,omitempty"`
	Error string    `json:"error,omitempty"`
}

// Append writes one NDJSON line and trims to keep.
func Append(path string, r Record, keep int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		f.Close()
		return err
	}
	_, werr := f.Write(append(b, '\n'))
	f.Close()
	if werr != nil {
		return werr
	}
	return trim(path, keep)
}

// Read returns records newest first, skipping unparseable lines rather than failing:
// one torn line must not make the whole history unreadable.
func Read(path string, limit int) ([]Record, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	lines := splitLines(b)
	var out []Record
	for i := len(lines) - 1; i >= 0; i-- {
		var r Record
		if json.Unmarshal(lines[i], &r) != nil {
			continue
		}
		out = append(out, r)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
}

// Savings counts what the short circuit actually bought. This number IS the product:
// without it, "it sometimes answers things itself" is a claim rather than a result.
type Savings struct {
	Total       int `json:"total"`
	Answered    int `json:"answered_locally"`
	Routed      int `json:"routed"`
	Passed      int `json:"passed_through"`
	ShadowWould int `json:"shadow_would_have_answered"`
}

func Count(records []Record) Savings {
	var s Savings
	for _, r := range records {
		s.Total++
		switch r.Plan.Stage {
		case StageAnswered:
			s.Answered++
		case StageRouted:
			s.Routed++
		default:
			s.Passed++
		}
		if sc := r.Plan.ShortCircuit; sc != nil && sc.Shadow && sc.Verdict == answer.VerdictAnswered {
			s.ShadowWould++
		}
	}
	return s
}

func splitLines(b []byte) [][]byte {
	var out [][]byte
	start := 0
	for i, c := range b {
		if c == '\n' {
			if i > start {
				out = append(out, b[start:i])
			}
			start = i + 1
		}
	}
	if start < len(b) {
		out = append(out, b[start:])
	}
	return out
}

func trim(path string, keep int) error {
	if keep <= 0 {
		return nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	lines := splitLines(b)
	if len(lines) <= keep {
		return nil
	}
	lines = lines[len(lines)-keep:]
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	for _, l := range lines {
		if _, err := f.Write(append(l, '\n')); err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
