// Package agentkind turns a tier and an effort into the arguments and environment an
// agent process is actually started with.
//
// This package is where the abstraction meets the CLI, and it is entirely config-driven
// for one reason: claude, codex, gemini and the rest name their models differently and
// take their flags differently, so a hardcoded haiku/sonnet/opus is a router that works
// for exactly one agent. The decision model stays blind to every name in here.
//
// HONESTY ABOUT WHAT HERDR GIVES US (verified against a live 0.9.x server, SPEC.md §3):
//
//   - `agent.start` takes `{name, kind, pane_id, args[]}` — so the agent's own CLI
//     flags ARE ours to choose at start. That is the model flag.
//   - `pane.split` takes `{cwd, env{}}` — so the process environment is ours to choose
//     at pane creation. That is how an effort setting with no flag gets through.
//   - NOTHING lets us change the model of an agent that is ALREADY RUNNING. There is no
//     hook, no interception, and no such request in the protocol.
//
// So routing happens at DISPATCH — a new pane, a new agent, chosen arguments — and
// never mid-session. Claiming otherwise would be claiming a capability Herdr does not
// have.
package agentkind

import (
	"fmt"
	"strings"

	"github.com/muthuishere/herdr-jev/internal/policy"
)

// Kind is one agent kind's mapping. It is config, and the defaults below are a starting
// point to be corrected per machine, not a claim about any vendor's current lineup.
type Kind struct {
	// Tiers maps fast/balanced/deep to this agent's model names.
	Tiers policy.Tiers `toml:"tiers" json:"tiers"`
	// ModelFlag is the flag that selects a model, e.g. "--model". Empty means this
	// agent takes no model flag and only ModelEnv applies.
	ModelFlag string `toml:"model_flag" json:"model_flag"`
	// ModelEnv sets the model through the environment instead of, or as well as, a
	// flag. pane.split carries env; that is the whole reason this field exists.
	ModelEnv string `toml:"model_env" json:"model_env"`
	// EffortFlag / EffortEnv are the same for reasoning effort. When BOTH are empty
	// this agent has no effort control we know of, and effort is simply not routed —
	// reported, not silently dropped.
	EffortFlag string `toml:"effort_flag" json:"effort_flag"`
	EffortEnv  string `toml:"effort_env" json:"effort_env"`
	// EffortValues remaps our ladder onto this agent's vocabulary when it differs.
	EffortValues map[string]string `toml:"effort_values" json:"effort_values"`
	// FamilyHints rank a model id that matches no configured tier name.
	FamilyHints map[string]int `toml:"family_hints" json:"family_hints"`
}

// Defaults are the kinds we ship with. Every one of them is overridable in config, and
// `herdr-jev doctor` prints what is in effect — because a flag that a vendor renamed is
// a silently unrouted agent, and the only defence is making the mapping visible.
func Defaults() map[string]Kind {
	return map[string]Kind{
		"claude": {
			Tiers:       policy.Tiers{Fast: "haiku", Balanced: "sonnet", Deep: "opus"},
			ModelFlag:   "--model",
			FamilyHints: map[string]int{"haiku": 0, "sonnet": 1, "opus": 2},
			// Claude Code takes no effort flag we can rely on across versions, so
			// effort is not routed for it by default rather than guessed at.
		},
		"codex": {
			Tiers:        policy.Tiers{Fast: "gpt-5-mini", Balanced: "gpt-5", Deep: "gpt-5"},
			ModelFlag:    "--model",
			EffortFlag:   "-c",
			EffortValues: map[string]string{"low": "low", "medium": "medium", "high": "high", "xhigh": "high"},
			FamilyHints:  map[string]int{"mini": 0, "gpt-5": 1},
		},
		"gemini": {
			Tiers:       policy.Tiers{Fast: "gemini-flash-lite", Balanced: "gemini-flash", Deep: "gemini-pro"},
			ModelFlag:   "--model",
			FamilyHints: map[string]int{"flash-lite": 0, "flash": 1, "pro": 2},
		},
	}
}

// Launch is everything needed to start an agent at a chosen tier and effort.
type Launch struct {
	Args []string          `json:"args"`
	Env  map[string]string `json:"env"`
	// Notes record what could NOT be applied and why. An effort we were asked for
	// and silently dropped would make the transcript a lie.
	Notes []string `json:"notes,omitempty"`
}

// Build turns a model name and an effort into process arguments and environment.
//
// Empty model or effort means "leave it to the agent's own default", which is exactly
// what a policy that declined to act wants — never a substituted guess.
func (k Kind) Build(model string, effort policy.Effort) Launch {
	l := Launch{Env: map[string]string{}}

	if model != "" {
		switch {
		case k.ModelFlag != "":
			l.Args = append(l.Args, k.ModelFlag, model)
		case k.ModelEnv != "":
			l.Env[k.ModelEnv] = model
		default:
			l.Notes = append(l.Notes, fmt.Sprintf("model %q not applied: this agent kind has neither model_flag nor model_env configured", model))
		}
		if k.ModelFlag != "" && k.ModelEnv != "" {
			l.Env[k.ModelEnv] = model
		}
	}

	if effort != "" {
		value := string(effort)
		if mapped, ok := k.EffortValues[value]; ok {
			value = mapped
		}
		switch {
		case k.EffortFlag == "-c":
			// codex's generic config override takes key=value as one argument.
			l.Args = append(l.Args, "-c", "model_reasoning_effort="+value)
		case k.EffortFlag != "":
			l.Args = append(l.Args, k.EffortFlag, value)
		case k.EffortEnv != "":
			l.Env[k.EffortEnv] = value
		default:
			l.Notes = append(l.Notes, fmt.Sprintf("effort %q not applied: this agent kind exposes no effort control we know of", effort))
		}
	}
	return l
}

// Known reports whether a kind name is one Herdr itself will start. Herdr's
// `agent.start` takes a closed set of kinds, so a typo here fails at the socket rather
// than at the terminal — but failing here, by name, is a far better error.
func Known(name string) bool {
	for _, k := range herdrKinds {
		if strings.EqualFold(k, name) {
			return true
		}
	}
	return false
}

// herdrKinds is the enum `herdr agent start --kind` accepts on 0.9.x.
var herdrKinds = []string{
	"pi", "claude", "codex", "gemini", "cursor", "devin", "agy", "cline", "omp",
	"mastracode", "opencode", "copilot", "kimi", "kiro", "droid", "amp", "grok",
	"hermes", "kilo", "qodercli", "qwen", "maki", "muse",
}

// HerdrKinds returns the accepted kinds, for error messages and doctor output.
func HerdrKinds() []string { return append([]string(nil), herdrKinds...) }
