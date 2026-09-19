// Package answer decides whether a task can be answered by the decision model itself,
// with no agent invoked at all, and — when it can — answers it.
//
// This is the short circuit, and it is the sharpest thing in the plugin. A
// decision-shaped task answered locally costs milliseconds, zero tokens, and sends
// nothing off the machine. The same short circuit aimed at a generative task produces
// a probability where code was wanted, which is worse than being slow.
//
// So the gate is in two parts, in this order, and the order is the design:
//
//  1. SHAPE, decided here, without the model, from the text alone.
//  2. CONFIDENCE, decided by the model, against its own high bar.
//
// A task that is not decision-shaped is never short-circuited no matter how confident
// the model is. Confidence cannot rescue the wrong question.
package answer

import (
	"regexp"
	"strings"
)

// Shape is what kind of question a task is, if it is one at all.
//
// A closed set on purpose. These are exactly the shapes an NLI cross-encoder answers
// natively — entailment between a premise and a hypothesis, and nothing else. Every
// new shape must be a new entailment framing, not a new hope.
type Shape string

const (
	// ShapeNone is the default and the safe one: hand it to an agent.
	ShapeNone Shape = "none"
	// ShapeBoolean is a yes/no or true/false question. Hypothesis = the claim.
	ShapeBoolean Shape = "boolean"
	// ShapePickOne is choose-one-of-N. Rerank over the options.
	ShapePickOne Shape = "pick_one"
	// ShapeGrade is answer-vs-reference. Only ever from explicit flags.
	ShapeGrade Shape = "grade"
)

// Detection is the verdict and, always, the reason for it. "Why was my question not
// answered locally" must be answerable without reading this source.
type Detection struct {
	Shape   Shape    `json:"shape"`
	Reason  string   `json:"reason"`
	Options []string `json:"options,omitempty"`
}

// generativeVerbs veto the short circuit outright.
//
// This list is the guardrail, and it is checked FIRST, before any shape is considered.
// "Should I use a mutex here, and if so write it" is a question with a yes/no clause in
// front of an implementation request; answering the clause and stopping would be a
// silent, confident failure to do the work. When a task contains generative intent at
// all, an agent runs it.
//
// Conservative by construction: a false veto costs one agent call, which is what would
// have happened anyway. A false pass costs the user their actual task.
var generativeVerbs = []string{
	"write", "edit", "add", "implement", "refactor", "fix", "create", "build",
	"generate", "make", "update", "change", "modify", "rename", "delete", "remove",
	"run", "execute", "install", "deploy", "migrate", "commit", "push", "merge",
	"debug", "investigate", "explain", "summarise", "summarize", "describe",
	"review", "draft", "design", "plan", "port", "translate", "rewrite", "test",
	"analyse", "analyze", "document", "set up", "clean up", "look into", "figure out",
}

// interrogativeLeads are the openers of a genuine yes/no question. A question mark
// alone is not enough — "how do I fix this?" has one and is not answerable here.
var interrogativeLeads = []string{
	"is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ",
	"should ", "would ", "will ", "has ", "have ", "had ", "am ", "isn't ",
	"aren't ", "doesn't ", "don't ", "didn't ", "shouldn't ", "must ",
}

var (
	// "A or B?" / "X, Y or Z?" — an explicit closed choice in the text.
	orChoice = regexp.MustCompile(`(?i)^(?:which|what)\b.*\bor\b.*\?$`)
	// A numbered or bulleted option list following a "which" question.
	listItem = regexp.MustCompile(`(?m)^\s*(?:[-*]|\(?\d+[.)])\s+(\S.*)$`)
	wordish  = regexp.MustCompile(`[a-z']+`)
)

// Detect classifies a task's shape from its text and any explicitly supplied options.
//
// explicitOptions come from `--option` flags: a caller that hands us a closed set has
// told us the shape directly, which is far stronger evidence than any phrasing, and is
// the only path that needs no interrogative form at all.
func Detect(task string, explicitOptions []string, gradeReference string) Detection {
	t := strings.ToLower(strings.TrimSpace(task))
	if t == "" {
		return Detection{ShapeNone, "empty task", nil}
	}

	// A grading request is explicit or it does not exist. Inferring "grade this"
	// from prose is how a code review becomes a probability.
	if gradeReference != "" {
		return Detection{ShapeGrade, "an explicit reference was supplied to grade against", nil}
	}

	// The veto, first and unconditionally. Word-boundary matched so "rewritten" in
	// a question about existing code does not veto, but "rewrite the parser" does.
	if verb, ok := hasGenerativeVerb(t); ok {
		return Detection{ShapeNone, "the task asks to " + verb + ", which is generative work an agent must do", nil}
	}

	if len(explicitOptions) >= 2 {
		return Detection{ShapePickOne, "the caller supplied a closed set of options", explicitOptions}
	}
	if len(explicitOptions) == 1 {
		// One option is not a choice. Refusing is better than silently comparing
		// it against nothing and reporting a number.
		return Detection{ShapeNone, "pick-one needs at least two options; one was given", nil}
	}

	if !strings.HasSuffix(t, "?") {
		return Detection{ShapeNone, "not phrased as a question", nil}
	}

	if opts := listOptions(task); len(opts) >= 2 && strings.HasPrefix(t, "which") {
		return Detection{ShapePickOne, "a 'which' question with a listed set of options", opts}
	}
	if orChoice.MatchString(t) {
		if opts := splitOr(task); len(opts) >= 2 {
			return Detection{ShapePickOne, "a closed choice written as 'A or B'", opts}
		}
	}

	for _, lead := range interrogativeLeads {
		if strings.HasPrefix(t, lead) {
			return Detection{ShapeBoolean, "a yes/no question opening with '" + strings.TrimSpace(lead) + "'", nil}
		}
	}
	if strings.HasPrefix(t, "true or false") {
		return Detection{ShapeBoolean, "an explicit true/false question", nil}
	}

	// Everything else — "why", "how", "what happened", an open "which" with no
	// options — is open-ended. Open-ended is an agent's job.
	return Detection{ShapeNone, "an open-ended question; only yes/no and pick-one are answerable here", nil}
}

// nounMarkers are the words that, immediately before a verb, mean it is being used as
// a NOUN.
//
// "Does the state file survive a port change?" is a question, not a request to change
// anything, and vetoing it would refuse to answer half the genuine questions anyone
// asks about a codebase — "change", "test", "review", "plan", "design" and "build" are
// all nouns at least as often as verbs. The veto stays broad everywhere else: a false
// veto costs one agent call, which is what would have happened anyway, so this is the
// only narrowing worth making.
var nounMarkers = map[string]bool{
	"a": true, "an": true, "the": true, "this": true, "that": true, "these": true,
	"those": true, "its": true, "our": true, "my": true, "your": true, "their": true,
	"each": true, "every": true, "any": true, "no": true, "one": true, "last": true,
	"next": true, "first": true, "same": true, "whole": true, "of": true, "in": true,
	"for": true, "after": true, "before": true, "during": true, "per": true,
}

func hasGenerativeVerb(lowered string) (string, bool) {
	words := wordish.FindAllString(lowered, -1)
	verbs := make(map[string]bool, len(generativeVerbs))
	for _, v := range generativeVerbs {
		if !strings.Contains(v, " ") {
			verbs[v] = true
		}
	}
	// nounPhrase carries across adjacent words: in "a port change" the determiner
	// marks "port" as a noun, and "port" then marks "change" as one too. Without the
	// carry, the second word of every compound noun reads as an imperative.
	nounPhrase := false
	for i, w := range words {
		marked := nounMarkers[w] || (nounPhrase && verbs[w])
		if verbs[w] {
			if i > 0 && (nounMarkers[words[i-1]] || nounPhrase) {
				nounPhrase = true
				continue // used as a noun: "a port change", "the test suite"
			}
			return w, true
		}
		nounPhrase = marked
	}
	for _, v := range generativeVerbs {
		if strings.Contains(v, " ") && strings.Contains(lowered, v) {
			return v, true
		}
	}
	return "", false
}

// listOptions pulls a bulleted or numbered set out of the task text.
func listOptions(task string) []string {
	var out []string
	for _, m := range listItem.FindAllStringSubmatch(task, -1) {
		if s := strings.TrimSpace(m[1]); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// splitOr turns "which is faster, a map or a slice?" into its alternatives.
//
// Deliberately crude, and that is safe: a bad split produces options that score
// alike, the confidence bar is not cleared, and the task falls through to an agent.
func splitOr(task string) []string {
	s := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(task), "?"))
	if i := strings.IndexAny(s, ",:"); i >= 0 {
		s = s[i+1:]
	} else if i := strings.Index(strings.ToLower(s), " "); i >= 0 {
		// No delimiter: drop the leading interrogative word only.
		s = s[i+1:]
	}
	parts := regexp.MustCompile(`(?i)\s*,\s*|\s+or\s+`).Split(s, -1)
	var out []string
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	if len(out) < 2 {
		return nil
	}
	return out
}
