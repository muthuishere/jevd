package agentkind

import (
	"strings"
	"testing"

	"github.com/muthuishere/herdr-jev/internal/policy"
)

func TestBuildProducesTheArgumentsTheAgentActuallyTakes(t *testing.T) {
	cases := []struct {
		name     string
		kind     Kind
		model    string
		effort   policy.Effort
		wantArgs string
		wantEnv  map[string]string
		wantNote string
	}{
		{
			name:     "a model flag becomes two arguments",
			kind:     Kind{ModelFlag: "--model"},
			model:    "opus",
			wantArgs: "--model opus",
		},
		{
			name:    "an agent with no flag uses the environment instead",
			kind:    Kind{ModelEnv: "AGENT_MODEL"},
			model:   "opus",
			wantEnv: map[string]string{"AGENT_MODEL": "opus"},
		},
		{
			name:     "codex's generic config override is one argument, not two",
			kind:     Kind{ModelFlag: "--model", EffortFlag: "-c"},
			model:    "gpt-5",
			effort:   "high",
			wantArgs: "--model gpt-5 -c model_reasoning_effort=high",
		},
		{
			name:     "our ladder is remapped onto the agent's own vocabulary",
			kind:     Kind{EffortFlag: "-c", EffortValues: map[string]string{"xhigh": "high"}},
			effort:   "xhigh",
			wantArgs: "-c model_reasoning_effort=high",
		},
		{
			// A silently dropped effort would make the transcript a lie: the log
			// would say it routed, and nothing would have changed.
			name:     "an agent with no effort control says so rather than dropping it",
			kind:     Kind{ModelFlag: "--model"},
			model:    "opus",
			effort:   "high",
			wantArgs: "--model opus",
			wantNote: "effort",
		},
		{
			name:     "an agent with no model control at all says so",
			kind:     Kind{},
			model:    "opus",
			wantNote: "model",
		},
		{
			// An empty model or effort means "leave it to the agent's own
			// default", which is exactly what a policy that declined wants.
			name: "nothing chosen means nothing set",
			kind: Kind{ModelFlag: "--model", EffortFlag: "--effort"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := c.kind.Build(c.model, c.effort)
			if args := strings.Join(got.Args, " "); args != c.wantArgs {
				t.Errorf("args = %q, want %q", args, c.wantArgs)
			}
			for k, v := range c.wantEnv {
				if got.Env[k] != v {
					t.Errorf("env[%s] = %q, want %q", k, got.Env[k], v)
				}
			}
			notes := strings.Join(got.Notes, " ")
			if c.wantNote == "" && notes != "" {
				t.Errorf("unexpected note: %s", notes)
			}
			if c.wantNote != "" && !strings.Contains(notes, c.wantNote) {
				t.Errorf("notes = %q, want one mentioning %q", notes, c.wantNote)
			}
		})
	}
}

func TestDefaultsAreAllKindsHerdrCanActuallyStart(t *testing.T) {
	// A kind herdr's `agent start` enum does not accept fails at the socket with an
	// opaque error. Failing here, by name, is a far better error — and shipping a
	// default that cannot start is simply a bug.
	for name := range Defaults() {
		if !Known(name) {
			t.Errorf("default agent kind %q is not one herdr can start", name)
		}
	}
}

func TestDefaultTiersAreCompleteAndDistinctWhereTheyMustBe(t *testing.T) {
	for name, k := range Defaults() {
		if k.Tiers.Fast == "" || k.Tiers.Balanced == "" || k.Tiers.Deep == "" {
			t.Errorf("%s has an empty tier: %+v", name, k.Tiers)
		}
		if k.ModelFlag == "" && k.ModelEnv == "" {
			t.Errorf("%s has no way to select a model, so it can never be routed", name)
		}
		// Fast must differ from deep, or there is no routing to do at all.
		if k.Tiers.Fast == k.Tiers.Deep {
			t.Errorf("%s maps fast and deep to the same model", name)
		}
	}
}
