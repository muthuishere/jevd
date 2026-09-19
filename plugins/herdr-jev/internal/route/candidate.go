package route

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/herdrapi"
)

// Candidate is one pane, distilled into the single string the cross-encoder ranks.
//
// Everything the model knows about a pane is Option. What goes in it decides every
// routing decision this plugin will ever make, so the rules are fixed in SPEC.md §3.2
// and covered by table-driven tests rather than tuned by feel.
type Candidate struct {
	Pane   herdrapi.Pane `json:"-"`
	Option string        `json:"-"`

	PaneID     string `json:"pane_id"`
	TerminalID string `json:"terminal_id"`
	Agent      string `json:"agent"`
	Status     string `json:"status"`
	Cwd        string `json:"cwd"`
	Title      string `json:"title"`
	OptionSHA  string `json:"option_sha"`
}

// Eligible decides who gets ranked at all.
//
// A pane that cannot be prompted must not appear in a ranking: offering a choice we
// cannot honour is how a confident score turns into a silent no-op. Returns the reason
// for exclusion, which `panes` prints — an invisible filter is a filter nobody can
// debug.
func Eligible(p herdrapi.Pane, selfPane string, r config.Routing) (bool, string) {
	if p.Agent == "" {
		return false, "no agent detected (a bare shell cannot be prompted)"
	}
	if p.PaneID != "" && p.PaneID == selfPane {
		// An agent that routes to itself answers itself and routes again.
		return false, "this is our own pane"
	}
	if p.Status == "blocked" {
		// Herdr rejects agent.prompt on a blocked agent before any input is sent,
		// so ranking it would be ranking an undeliverable option.
		return false, "agent is blocked; a human is needed there first"
	}
	if !r.IncludeWorking && p.Status == "working" {
		return false, "agent is working and routing.include_working is false"
	}
	for _, a := range r.ExcludeAgents {
		if strings.EqualFold(a, p.Agent) {
			return false, "agent " + p.Agent + " is in routing.exclude_agents"
		}
	}
	for _, w := range r.ExcludeWorkspaces {
		if w == p.WorkspaceID {
			return false, "workspace " + w + " is in routing.exclude_workspaces"
		}
	}
	return true, ""
}

// Distil turns a pane and its recent output into the one option string the model sees.
//
// Identity first, output last, because truncation eats the tail and the parts we are
// surest about must survive it.
func Distil(p herdrapi.Pane, output string, r config.Routing, maxField int) Candidate {
	var b strings.Builder

	agent := p.Agent
	if agent == "" {
		agent = "shell"
	}
	fmt.Fprintf(&b, "agent %s", agent)
	if cwd := shortCwd(p.Cwd); cwd != "" {
		fmt.Fprintf(&b, " in %s", cwd)
	}
	status := p.Status
	if status == "" {
		status = "unknown"
	}
	fmt.Fprintf(&b, ", status %s.", status)

	title := strings.TrimSpace(firstNonEmpty(p.Label, p.Title))
	if title != "" {
		fmt.Fprintf(&b, "\nTitle: %s", collapse(title))
	}

	if clean := cleanOutput(output, r.OutputLines, r.MaxOutputChars); clean != "" {
		fmt.Fprintf(&b, "\nRecent output: %s", clean)
	}

	option := truncate(b.String(), maxField)
	sum := sha256.Sum256([]byte(option))

	return Candidate{
		Pane:       p,
		Option:     option,
		PaneID:     p.PaneID,
		TerminalID: p.TerminalID,
		Agent:      agent,
		Status:     status,
		Cwd:        shortCwd(p.Cwd),
		Title:      title,
		OptionSHA:  hex.EncodeToString(sum[:8]),
	}
}

// shortCwd keeps the basename and one parent.
//
// The rest is /Users/<me>/muthu/gitworkspace on every single pane — tokens that are
// identical across every option and therefore carry exactly zero ranking signal while
// costing field budget.
func shortCwd(cwd string) string {
	cwd = strings.TrimSpace(cwd)
	if cwd == "" {
		return ""
	}
	cwd = strings.TrimRight(cwd, string(filepath.Separator))
	base := filepath.Base(cwd)
	parent := filepath.Base(filepath.Dir(cwd))
	if parent == "" || parent == "." || parent == string(filepath.Separator) || parent == base {
		return base
	}
	return parent + "/" + base
}

// noiseLine reports whether a line is terminal furniture rather than content.
//
// Organised against: a pane sitting on a spinner or a box-drawn frame out-ranking a
// pane whose output is genuinely about the message. Decoration is not signal, and a
// cross-encoder cannot tell the difference — so we do it here.
func noiseLine(line string) bool {
	t := strings.TrimSpace(line)
	if t == "" {
		return true
	}
	stripped := strings.TrimFunc(t, func(r rune) bool {
		switch {
		case r >= 0x2500 && r <= 0x257F: // box drawing
			return true
		case r >= 0x2580 && r <= 0x259F: // block elements (progress bars)
			return true
		case r >= 0x25A0 && r <= 0x25FF: // geometric shapes (spinner frames)
			return true
		case r >= 0x2800 && r <= 0x28FF: // braille (the other spinner alphabet)
			return true
		case r == ' ' || r == '\t' || r == '·' || r == '-' || r == '=' || r == '_':
			return true
		}
		return false
	})
	return stripped == ""
}

// promptEcho reports whether a line is the agent's own input box.
//
// Organised against STICKINESS, the subtlest failure here: an agent's prompt box often
// still holds the LAST message routed to it, so leaving it in makes every message look
// like it belongs wherever the previous one went. The router then looks like it is
// working — it is confident, consistent, and wrong.
func promptEcho(line string) bool {
	t := strings.TrimSpace(line)
	t = strings.TrimLeft(t, "│|┃ \t")
	t = strings.TrimSpace(t)
	for _, p := range []string{">", "❯", "»", "$", "✻", "✽", "✢"} {
		if strings.HasPrefix(t, p) {
			return true
		}
	}
	return false
}

// cleanOutput collapses a scrollback into the tail that carries signal.
//
// The cap is not decoration: an uncapped 100k-line scrollback swamps the identity line
// that truncation is supposed to protect, and blows the server's max_field_chars.
func cleanOutput(raw string, maxLines, maxChars int) string {
	if raw == "" || maxChars == 0 {
		return ""
	}
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	var kept []string
	for _, l := range lines {
		if noiseLine(l) || promptEcho(l) {
			continue
		}
		kept = append(kept, strings.Join(strings.Fields(l), " "))
	}
	if maxLines > 0 && len(kept) > maxLines {
		kept = kept[len(kept)-maxLines:] // the tail is the recent part
	}
	out := strings.Join(kept, " ")
	if maxChars > 0 && len(out) > maxChars {
		// Keep the END: the most recent output is the most diagnostic of what
		// this agent is doing right now.
		out = out[len(out)-maxChars:]
	}
	return strings.TrimSpace(out)
}

func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max]
}

func collapse(s string) string { return strings.Join(strings.Fields(s), " ") }

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
