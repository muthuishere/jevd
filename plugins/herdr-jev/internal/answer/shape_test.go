package answer

import "testing"

// Shape detection is the crux of the whole feature: answering a generative task with a
// probability is worse than being slow, and no confidence can rescue the wrong
// question. So the table below is deliberately heavy on the things that must NOT be
// short-circuited.
func TestDetect(t *testing.T) {
	cases := []struct {
		name string
		task string
		opts []string
		ref  string
		want Shape
	}{
		// --- must never short-circuit: generative work -----------------------
		{"write", "Write a parser for this grammar", nil, "", ShapeNone},
		{"refactor", "Refactor the router into smaller functions", nil, "", ShapeNone},
		{"fix", "Fix the flaky rerank test", nil, "", ShapeNone},
		{"explain", "Explain how the confidence bars work", nil, "", ShapeNone},
		{"debug", "Debug why the daemon exits at boot", nil, "", ShapeNone},
		{"run", "Run the test suite and report", nil, "", ShapeNone},
		{"deploy", "Deploy this to prod", nil, "", ShapeNone},
		{"summarise", "Summarise this file", nil, "", ShapeNone},

		// The veto beats the question form. This is the case the gate exists for:
		// answering the yes/no clause and stopping would silently skip the work.
		{"question wrapping an implementation request",
			"Should we use a mutex here, and if so write it?", nil, "", ShapeNone},
		{"question asking to change something",
			"Can you update the timeout to 30s?", nil, "", ShapeNone},

		// --- open-ended is an agent's job -----------------------------------
		{"how", "How do I make this faster?", nil, "", ShapeNone},
		{"why", "Why does the pane read come back empty?", nil, "", ShapeNone},
		{"what happened", "What happened in the last deploy?", nil, "", ShapeNone},
		{"which with no options", "Which approach is better?", nil, "", ShapeNone},
		{"no question mark", "is the build green", nil, "", ShapeNone},
		{"empty", "", nil, "", ShapeNone},

		// --- genuinely decision-shaped --------------------------------------
		{"is", "Is the openjev API version 1 compatible with this client?", nil, "", ShapeBoolean},
		{"does", "Does the state file survive a port change?", nil, "", ShapeBoolean},
		{"should", "Should the downgrade bar be higher than the upgrade bar?", nil, "", ShapeBoolean},
		{"true or false", "True or false: pane ids are stable across restarts?", nil, "", ShapeBoolean},
		{"explicit options", "which backend answers this", []string{"openjev", "typesafe"}, "", ShapePickOne},
		{"explicit grade", "the floor is 0.55", nil, "the confidence floor defaults to 0.55", ShapeGrade},

		// --- refusals that protect the caller from a meaningless number ------
		{"one option is not a choice", "which one", []string{"only"}, "", ShapeNone},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Detect(c.task, c.opts, c.ref)
			if got.Shape != c.want {
				t.Errorf("Detect(%q) = %s (%s), want %s", c.task, got.Shape, got.Reason, c.want)
			}
			if got.Reason == "" {
				t.Error("every detection must carry a reason; 'why was this not answered locally' has to be answerable")
			}
		})
	}
}

func TestDetectPicksOneFromAWrittenChoice(t *testing.T) {
	d := Detect("Which is faster, a map or a slice?", nil, "")
	if d.Shape != ShapePickOne {
		t.Fatalf("shape = %s (%s)", d.Shape, d.Reason)
	}
	if len(d.Options) != 2 {
		t.Fatalf("options = %v, want two alternatives", d.Options)
	}
}

func TestExplicitOptionsBeatPhrasing(t *testing.T) {
	// A caller handing us a closed set has told us the shape directly, which is far
	// stronger evidence than any wording — and is the only path needing no
	// interrogative form at all.
	d := Detect("pick the right pane", []string{"a", "b", "c"}, "")
	if d.Shape != ShapePickOne {
		t.Fatalf("shape = %s (%s)", d.Shape, d.Reason)
	}
}

func TestGenerativeVerbVetoBeatsExplicitOptions(t *testing.T) {
	// Even a closed set does not license answering "write one of these".
	d := Detect("write one of these", []string{"a", "b"}, "")
	if d.Shape != ShapeNone {
		t.Fatalf("shape = %s, want none", d.Shape)
	}
}
