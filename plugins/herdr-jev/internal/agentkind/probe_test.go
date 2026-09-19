package agentkind

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/muthuishere/herdr-jev/internal/policy"
)

// fakeBin writes an executable that prints the given help text, and puts it on PATH.
// Probing real vendor CLIs in a test would make the suite depend on what happens to be
// installed, which is the opposite of a regression net.
func fakeBin(t *testing.T, name, help string) {
	t.Helper()
	dir := t.TempDir()
	script := "#!/bin/sh\ncat <<'HELP'\n" + help + "\nHELP\n"
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func find(fs []Finding, what string) *Finding {
	for i := range fs {
		if fs[i].What == what {
			return &fs[i]
		}
	}
	return nil
}

func TestProbeReportsThreeStatesAndNeverPassesAnUnknown(t *testing.T) {
	cases := []struct {
		name  string
		help  string
		kind  Kind
		what  string
		state State
	}{
		{
			// The failure the probe exists for: a vendor renames the flag, and the
			// plugin goes on logging confident decisions above arguments nothing
			// accepts.
			name:  "a flag the binary does not document is REJECTED",
			help:  "Usage: agent [--profile NAME]\n  --profile   which profile to use",
			kind:  Kind{ModelFlag: "--model", Tiers: policy.Tiers{Fast: "a", Balanced: "b", Deep: "c"}},
			what:  "model_flag",
			state: Rejected,
		},
		{
			name:  "a documented flag is ACCEPTED",
			help:  "Usage: agent [--model NAME]\n  --model   which model to use",
			kind:  Kind{ModelFlag: "--model", Tiers: policy.Tiers{Fast: "a", Balanced: "b", Deep: "c"}},
			what:  "model_flag",
			state: Accepted,
		},
		{
			// Help text lists flags and almost never enumerates models, so an
			// absent model name is NOT evidence the binary refuses it. Calling
			// that REJECTED would cry wolf on every correctly configured agent.
			name:  "a model name absent from the help is UNKNOWN, never rejected",
			help:  "Usage: agent [--model NAME]",
			kind:  Kind{ModelFlag: "--model", Tiers: policy.Tiers{Fast: "zeta", Balanced: "b", Deep: "c"}},
			what:  "tiers.fast",
			state: Unknown,
		},
		{
			name:  "a model name the help does list is ACCEPTED",
			help:  "Usage: agent [--model NAME]\n  models: zeta, b, c",
			kind:  Kind{ModelFlag: "--model", Tiers: policy.Tiers{Fast: "zeta", Balanced: "b", Deep: "c"}},
			what:  "tiers.fast",
			state: Accepted,
		},
		{
			name:  "an unconfigured tier is REJECTED: that one IS our fault",
			help:  "Usage: agent [--model NAME]",
			kind:  Kind{ModelFlag: "--model", Tiers: policy.Tiers{Balanced: "b", Deep: "c"}},
			what:  "tiers.fast",
			state: Rejected,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fakeBin(t, "probetest", c.help)
			c.kind.Bin = "probetest"
			fs := Probe(context.Background(), "probetest", c.kind, 5*time.Second)
			got := find(fs, c.what)
			if got == nil {
				t.Fatalf("no finding for %q in %+v", c.what, fs)
			}
			if got.State != c.state {
				t.Errorf("%s = %s (%s), want %s", c.what, got.State, got.Detail, c.state)
			}
			if got.State != Accepted && got.Fix == "" {
				t.Error("every non-pass finding must name its own fix")
			}
		})
	}
}

func TestProbeOfAnAbsentBinaryIsUnknownNotAFailure(t *testing.T) {
	// "This agent is not installed here" is not something to fix in config, and
	// reporting it as a failure would train people to ignore the probe's output.
	fs := Probe(context.Background(), "definitely-not-installed-xyz", Kind{ModelFlag: "--model"}, time.Second)
	if len(fs) != 1 || fs[0].State != Unknown {
		t.Fatalf("want a single UNKNOWN finding, got %+v", fs)
	}
	if fs[0].Fix == "" {
		t.Error("even an UNKNOWN names what would let it be verified")
	}
}

func TestProbeIsBoundedByItsTimeout(t *testing.T) {
	// A binary that hangs must not hang the check. Without the deadline, one wedged
	// agent CLI makes `doctor --probe` look broken.
	dir := t.TempDir()
	path := filepath.Join(dir, "hangy")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nsleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))

	start := time.Now()
	fs := Probe(context.Background(), "hangy", Kind{Bin: "hangy", ModelFlag: "--model"}, 300*time.Millisecond)
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("probe took %s; the timeout is not bounding it", elapsed)
	}
	if len(fs) != 1 || fs[0].State != Unknown {
		t.Fatalf("a timed-out probe must be UNKNOWN, got %+v", fs)
	}
}

func TestProbeReadsHelpFromABinaryThatExitsNonZero(t *testing.T) {
	// Several of these CLIs print help and exit 1. Discarding that output would turn
	// every one of them into a false UNKNOWN.
	dir := t.TempDir()
	path := filepath.Join(dir, "grumpy")
	if err := os.WriteFile(path, []byte("#!/bin/sh\necho '  --model NAME'\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))

	fs := Probe(context.Background(), "grumpy", Kind{Bin: "grumpy", ModelFlag: "--model",
		Tiers: policy.Tiers{Fast: "a", Balanced: "b", Deep: "c"}}, 5*time.Second)
	if f := find(fs, "model_flag"); f == nil || f.State != Accepted {
		t.Fatalf("want the flag ACCEPTED despite exit 1, got %+v", fs)
	}
}
