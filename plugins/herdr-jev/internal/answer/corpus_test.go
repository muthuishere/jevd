package answer

import "testing"

// The corpus is the regression net for this feature's riskiest judgement.
//
// The `change` bug — "Does the state file survive a port change?" vetoed because
// `change` parsed as a verb — was found by an ad-hoc case, and it was luck that the case
// existed. Ad-hoc examples test the rules you already thought of; a corpus of real
// prompts tests the rules you did not.
//
// Both classes are here deliberately, and the generative half is the important half: a
// false veto costs one agent call, a false pass costs the user their actual task. If a
// change to shape.go makes any GENERATIVE row pass, that change is wrong, whatever else
// it improves.

// generative: must NEVER be short-circuited, at any confidence.
var generativeCorpus = []string{
	"Write a parser for the herdr plugin manifest",
	"Refactor the dispatcher into smaller functions",
	"Fix the flaky rerank test",
	"Add pagination to the users endpoint",
	"Implement the confidence floor",
	"Debug why the daemon exits immediately at boot",
	"Explain how the asymmetric bars work",
	"Summarise what changed in this diff",
	"Review my changes to the policy package",
	"Run the test suite and tell me what fails",
	"Deploy the crypto desk to prod",
	"Migrate the payments table to the new schema",
	"Rename Reranker to Backend everywhere",
	"Delete the archived worktrees",
	"Update the README to mention doctor --probe",
	"Design a sharding scheme for the events table",
	"Port the TypeScript policy to Go",
	"Set up a systemd unit for openjev",
	"Look into why routing is slow",
	"Figure out why the pane read comes back empty",
	// Question-shaped, but with generative intent riding along. Answering the clause
	// and stopping would be a silent, confident failure to do the work.
	"Should we use a mutex here, and if so write it?",
	"Can you update the timeout to 30s?",
	"Is this slow, and can you profile it?",
	// Open-ended: a question mark is necessary but nowhere near sufficient.
	"How do I make the classifier faster?",
	"Why does the pane read come back empty?",
	"What happened in the last deploy?",
	"What is the best way to structure this package?",
	"Which approach should I take here?",
}

// decisionShaped: must be recognised, so the short circuit gets its chance.
var decisionCorpus = []struct {
	prompt string
	want   Shape
}{
	{"Is the openjev API version compatible with this client?", ShapeBoolean},
	{"Does the state file survive a port change?", ShapeBoolean},
	{"Are pane ids stable across server restarts?", ShapeBoolean},
	{"Did the last deploy succeed?", ShapeBoolean},
	{"Should the downgrade bar be higher than the upgrade bar?", ShapeBoolean},
	{"Has the model finished downloading?", ShapeBoolean},
	{"Will this request exceed the max_options limit?", ShapeBoolean},
	{"Was this pane started by the plugin?", ShapeBoolean},
	{"Do these two configs agree on the answer bar?", ShapeBoolean},
	{"Must the pane be at a shell prompt for agent.start?", ShapeBoolean},
	{"Is 0.85 above the downgrade bar?", ShapeBoolean},
	{"True or false: the short circuit is on by default?", ShapeBoolean},
	// The noun-marker rule earning its place: "the test" is a noun phrase, so the
	// question survives the veto that "test the parser" would trigger.
	{"Is this the test that keeps failing?", ShapeBoolean},
	{"Which is faster, a map or a slice?", ShapePickOne},
	{"Which is cheaper, an extra rerank call or a second round trip?", ShapePickOne},
}

// knownConservativeVetoes are genuinely decision-shaped questions that the lexical gate
// vetoes anyway, because a generative verb appears in them as an ordinary word.
//
// These are a COST, not a bug, and they are written down rather than quietly tolerated.
// Telling "can herdr change a model?" (a question about a capability) from "can you
// change the model?" (a request) needs grammar this gate does not have, and the fragile
// heuristics that would fake it are exactly how a false PASS eventually slips through —
// which is the expensive mistake. Paying one agent call for each of these is the right
// side of that trade.
//
// If someone teaches the gate real grammar, these move up into decisionCorpus and this
// table shrinks. Until then it is the honest list of what we give up.
var knownConservativeVetoes = []string{
	"Can herdr change a running agent's model?",
	"Does the plugin write the config file?",
}

func TestKnownConservativeVetoesAreVetoedAndThatIsTheAcceptedCost(t *testing.T) {
	for _, p := range knownConservativeVetoes {
		t.Run(p, func(t *testing.T) {
			if d := Detect(p, nil, ""); d.Shape != ShapeNone {
				t.Errorf("shape = %s — this now passes the gate; move it into decisionCorpus and note what changed", d.Shape)
			}
		})
	}
}

func TestCorpusGenerativePromptsAreNeverShortCircuited(t *testing.T) {
	for _, p := range generativeCorpus {
		t.Run(p, func(t *testing.T) {
			d := Detect(p, nil, "")
			if d.Shape != ShapeNone {
				t.Errorf("shape = %s (%s)\nthis prompt must reach an agent; answering it with a probability is the failure the gate exists for",
					d.Shape, d.Reason)
			}
		})
	}
}

func TestCorpusDecisionShapedPromptsAreRecognised(t *testing.T) {
	for _, c := range decisionCorpus {
		t.Run(c.prompt, func(t *testing.T) {
			d := Detect(c.prompt, nil, "")
			if d.Shape != c.want {
				t.Errorf("shape = %s (%s), want %s", d.Shape, d.Reason, c.want)
			}
		})
	}
}

// A guard on the corpus itself: a net with holes in it looks exactly like a net.
func TestCorpusIsBigEnoughAndBalancedTowardsTheRiskierClass(t *testing.T) {
	if total := len(generativeCorpus) + len(decisionCorpus); total < 30 {
		t.Errorf("corpus has %d prompts; it is the regression net for the riskiest judgement here and should not shrink", total)
	}
	if len(generativeCorpus) <= len(decisionCorpus) {
		t.Error("the generative half must stay the larger one: a false pass costs the user their task, a false veto costs one agent call")
	}
}
