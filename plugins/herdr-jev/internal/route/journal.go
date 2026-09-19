package route

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Entry is one journalled routing decision — the unit `why` reads and the unit the
// `feed` pane renders.
//
// Scores for EVERY candidate are recorded, not just the winner's. A decision you
// cannot reconstruct afterwards is a decision you cannot tune, and an untunable router
// is one that gets switched off the first time it surprises somebody.
type Entry struct {
	V        int       `json:"v"`
	ID       string    `json:"id"`
	TS       time.Time `json:"ts"`
	Message  string    `json:"message"`
	Outcome  Outcome   `json:"outcome"`
	Reason   string    `json:"reason"`
	Floor    float64   `json:"floor"`
	Margin   float64   `json:"margin"`
	DryRun   bool      `json:"dry_run,omitempty"`
	Target   *Target   `json:"target,omitempty"`
	Ranked   []Ranked  `json:"candidates"`
	Excluded []Skipped `json:"excluded,omitempty"`
	Error    string    `json:"error,omitempty"`
}

// Target records BOTH ids. terminal_id is the stable one and is what a later resolve
// re-resolves through; pane_id is recorded only so the journal reads like what you saw
// at the time.
type Target struct {
	PaneID     string `json:"pane_id"`
	TerminalID string `json:"terminal_id"`
	Agent      string `json:"agent"`
}

// Skipped is a pane that was not ranked, and why. An invisible filter is a filter
// nobody can debug, and "why wasn't my pane considered" is the second question anyone
// asks after "why did it go there".
type Skipped struct {
	PaneID string `json:"pane_id"`
	Agent  string `json:"agent"`
	Reason string `json:"reason"`
}

// Hold is a message the router refused to guess about, waiting for a human.
//
// A hold is not a failure and not a drop. It carries its full ranking, so resolving it
// is picking from a list rather than retyping the message into a pane.
type Hold struct {
	ID      string    `json:"id"`
	TS      time.Time `json:"ts"`
	Message string    `json:"message"`
	Reason  string    `json:"reason"`
	Ranked  []Ranked  `json:"candidates"`
}

// NewID is a short, sortable-enough id. Time-prefixed so `hold list` and the journal
// read in order without a sort.
func NewID(prefix string) string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return fmt.Sprintf("%s-%s%s", prefix, time.Now().UTC().Format("150405"), hex.EncodeToString(b[:2]))
}

// AppendJournal writes one NDJSON line and trims the file to `keep` entries.
//
// Append-only with a trim, not a database: the journal must survive a crash mid-write
// without taking the previous 499 decisions with it, and one partial line at the end of
// an NDJSON file costs exactly one decision.
func AppendJournal(path string, e Entry, keep int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	b, err := json.Marshal(e)
	if err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return trim(path, keep)
}

// ReadJournal returns entries newest first, skipping unparseable lines rather than
// failing: one torn line must not make `why` useless for every other decision.
func ReadJournal(path string, limit int) ([]Entry, error) {
	lines, err := readLines(path)
	if err != nil {
		return nil, err
	}
	var out []Entry
	for i := len(lines) - 1; i >= 0; i-- {
		var e Entry
		if json.Unmarshal([]byte(lines[i]), &e) != nil {
			continue
		}
		out = append(out, e)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
}

// Holds reads the pending holds, oldest first — the order a human wants to work them.
func Holds(path string) ([]Hold, error) {
	lines, err := readLines(path)
	if err != nil {
		return nil, err
	}
	var out []Hold
	for _, l := range lines {
		var h Hold
		if json.Unmarshal([]byte(l), &h) == nil && h.ID != "" {
			out = append(out, h)
		}
	}
	return out, nil
}

func AppendHold(path string, h Hold) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(h)
	if err != nil {
		return err
	}
	_, err = f.Write(append(b, '\n'))
	return err
}

// RemoveHold drops one hold by id, rewriting the file.
//
// It is called only AFTER a resolve has actually delivered. Removing first would lose
// the message if delivery then failed, and a router that eats messages is worse than
// one that misroutes them.
func RemoveHold(path, id string) error {
	holds, err := Holds(path)
	if err != nil {
		return err
	}
	var kept []Hold
	found := false
	for _, h := range holds {
		if h.ID == id {
			found = true
			continue
		}
		kept = append(kept, h)
	}
	if !found {
		return fmt.Errorf("no held message with id %q", id)
	}
	return writeHolds(path, kept)
}

func writeHolds(path string, holds []Hold) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	for _, h := range holds {
		b, err := json.Marshal(h)
		if err != nil {
			f.Close()
			return err
		}
		if _, err := f.Write(append(b, '\n')); err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path) // atomic: a crash leaves the old list, never half of one
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		if len(sc.Bytes()) > 0 {
			out = append(out, sc.Text())
		}
	}
	return out, sc.Err()
}

func trim(path string, keep int) error {
	if keep <= 0 {
		return nil
	}
	lines, err := readLines(path)
	if err != nil || len(lines) <= keep {
		return err
	}
	lines = lines[len(lines)-keep:]
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	for _, l := range lines {
		if _, err := f.WriteString(l + "\n"); err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
