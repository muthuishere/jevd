package route

import (
	"strings"
	"testing"

	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/herdrapi"
)

func routing() config.Routing { return config.Default().Routing }

func TestEligibleExcludesWhatCannotBePrompted(t *testing.T) {
	cases := []struct {
		name string
		pane herdrapi.Pane
		self string
		ok   bool
		why  string
	}{
		{"a bare shell cannot be prompted",
			herdrapi.Pane{PaneID: "w1:p1"}, "", false, "no agent"},
		{"our own pane would be a self-loop",
			herdrapi.Pane{PaneID: "w1:p1", Agent: "claude"}, "w1:p1", false, "own pane"},
		{"a blocked agent would reject the prompt anyway",
			herdrapi.Pane{PaneID: "w1:p2", Agent: "claude", Status: "blocked"}, "", false, "blocked"},
		{"a working agent is still a candidate",
			herdrapi.Pane{PaneID: "w1:p2", Agent: "claude", Status: "working"}, "", true, ""},
		{"an idle agent is a candidate",
			herdrapi.Pane{PaneID: "w1:p3", Agent: "codex", Status: "idle"}, "", true, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ok, why := Eligible(c.pane, c.self, routing())
			if ok != c.ok {
				t.Fatalf("eligible = %v (%s), want %v", ok, why, c.ok)
			}
			if !ok && !strings.Contains(why, c.why) {
				t.Errorf("reason %q should mention %q", why, c.why)
			}
			if !ok && why == "" {
				t.Error("an invisible filter is a filter nobody can debug")
			}
		})
	}
}

func TestDistilPutsIdentityFirstSoTruncationEatsTheOutput(t *testing.T) {
	p := herdrapi.Pane{
		PaneID: "w1:p1", Agent: "claude", Status: "working",
		Cwd:   "/Users/someone/muthu/gitworkspace/my-jev",
		Title: "OpenJEV CLI executable server",
	}
	c := Distil(p, strings.Repeat("noise line\n", 500), routing(), 32768)

	if !strings.HasPrefix(c.Option, "agent claude in gitworkspace/my-jev, status working.") {
		t.Fatalf("identity must lead the option string, got:\n%s", c.Option)
	}
	if !strings.Contains(c.Option, "OpenJEV CLI executable server") {
		t.Error("the title carries real signal and must survive")
	}
	if len(c.Option) > routing().MaxOutputChars+400 {
		t.Errorf("option is %d chars; the output cap must bound it", len(c.Option))
	}
}

func TestDistilDropsNoiseAndPromptEcho(t *testing.T) {
	cases := []struct {
		name   string
		output string
		absent string
	}{
		// A pane sitting on a spinner must not out-rank one whose output is about
		// the message. Decoration is not signal.
		{"box drawing", "╭──────────────╮\n│              │\n╰──────────────╯\nreal content here", "─"},
		{"braille spinner frames", "⠋\n⠙\n⠹\nreal content here", "⠹"},
		{"blank and rule lines", "\n\n=====\n-----\nreal content here", "====="},
		// The stickiest failure available: an agent's prompt box still holds the
		// LAST message routed there, so leaving it in makes the router route to
		// wherever it last routed — confidently, consistently, and wrongly.
		{"prompt echo", "> fix the flaky rerank test\nreal content here", "flaky rerank"},
		{"boxed prompt echo", "│ > deploy to prod\nreal content here", "deploy to prod"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := cleanOutput(c.output, 40, 1200)
			if strings.Contains(got, c.absent) {
				t.Errorf("cleaned output still contains %q: %q", c.absent, got)
			}
			if !strings.Contains(got, "real content here") {
				t.Errorf("real content was dropped: %q", got)
			}
		})
	}
}

func TestShortCwdKeepsOnlyWhatDistinguishesPanes(t *testing.T) {
	// The full path is /Users/<me>/... on every pane: identical tokens carrying zero
	// ranking signal while costing field budget.
	cases := map[string]string{
		"/Users/me/muthu/gitworkspace/my-jev": "gitworkspace/my-jev",
		"/Users/me/my-jev/":                   "me/my-jev",
		"/my-jev":                             "my-jev",
		"":                                    "",
	}
	for in, want := range cases {
		if got := shortCwd(in); got != want {
			t.Errorf("shortCwd(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDecideHoldsRatherThanGuesses(t *testing.T) {
	c := []Candidate{
		{PaneID: "w1:p1", Agent: "claude"},
		{PaneID: "w1:p2", Agent: "codex"},
	}
	cases := []struct {
		name   string
		scores map[int]float64
		want   Outcome
	}{
		{"a clear winner is delivered", map[int]float64{0: 0.91, 1: 0.10}, OutcomeDelivered},
		{"below the floor it holds", map[int]float64{0: 0.41, 1: 0.38}, OutcomeHeld},
		// A floor alone is not enough: 0.86 vs 0.85 clears any sane floor and is
		// still a coin flip wearing a number. Near-ties are where misroutes live.
		{"a near tie holds even at high scores", map[int]float64{0: 0.86, 1: 0.85}, OutcomeHeld},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Decide(c, tc.scores, 0.55, 0.10)
			if d.Outcome != tc.want {
				t.Errorf("outcome = %s (%s), want %s", d.Outcome, d.Reason, tc.want)
			}
			if len(d.Ranked) != 2 {
				t.Error("every candidate's score must be kept, not just the winner's")
			}
		})
	}

	if got := Decide(nil, nil, 0.55, 0.10).Outcome; got != OutcomeNoCandidates {
		t.Errorf("no candidates = %s, want %s (not a hold: there is nobody to ask about)", got, OutcomeNoCandidates)
	}
	if got := Force(Decide(c, map[int]float64{0: 0.41, 1: 0.38}, 0.55, 0.10)); got.Outcome != OutcomeForced {
		t.Errorf("--force = %s, want forced", got.Outcome)
	}
}

func TestEqualScoresRankDeterministically(t *testing.T) {
	// A ranking that reshuffles between identical runs is a ranking nobody can debug.
	c := []Candidate{{PaneID: "w1:p9"}, {PaneID: "w1:p1"}}
	for i := 0; i < 20; i++ {
		d := Decide(c, map[int]float64{0: 0.5, 1: 0.5}, 0.4, 0.0)
		if d.Ranked[0].PaneID != "w1:p1" {
			t.Fatalf("run %d put %s first", i, d.Ranked[0].PaneID)
		}
	}
}
