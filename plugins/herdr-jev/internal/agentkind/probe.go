package agentkind

import (
	"context"
	"os/exec"
	"strings"
	"time"

	"github.com/muthuishere/herdr-jev/internal/policy"
)

// State is what a probe could establish about one mapping. Three values, not two.
//
// UNKNOWN is the whole reason this type exists. A mapping we could not verify must read
// DIFFERENTLY from one we verified — rendering "no binary on PATH" as a pass would
// reproduce, with a tick beside it, exactly the false confidence the probe was built to
// remove.
type State string

const (
	Accepted State = "ACCEPTED"
	Rejected State = "REJECTED"
	Unknown  State = "UNKNOWN"
)

// Finding is one probed fact and, when it is not a pass, what to do about it.
type Finding struct {
	Kind   string `json:"kind"`
	What   string `json:"what"`
	State  State  `json:"state"`
	Detail string `json:"detail"`
	Fix    string `json:"fix,omitempty"`
}

// Probe is a bounded, opt-in check that a kind's configured flags and model names are
// ones its binary actually knows about.
//
// WHY IT READS `--help` RATHER THAN TRYING THE FLAG. The direct test — run
// `<bin> --model opus --help` and see if it errors — is not available to us: for several
// of these binaries that starts an interactive agent, and for others it opens a session
// that costs tokens. A check that might launch an agent is not a check anyone will run.
// So we read the binary's own help text, which is cheap, side-effect free, and the same
// source the vendor documents the flag in.
//
// The cost of that choice is honest and bounded: help text reliably documents FLAGS and
// almost never enumerates MODEL NAMES. So a flag can be ACCEPTED or REJECTED, while an
// unlisted model name is UNKNOWN — never REJECTED, because "the help does not list it"
// is not evidence the binary refuses it.
func Probe(ctx context.Context, name string, k Kind, timeout time.Duration) []Finding {
	bin := k.Bin
	if bin == "" {
		bin = name // herdr calls the kind after its canonical executable
	}

	path, err := exec.LookPath(bin)
	if err != nil {
		// Not installed here is not a misconfiguration — it is simply unverifiable
		// on this machine, which is exactly what UNKNOWN is for.
		return []Finding{{name, "binary", Unknown, bin + " is not on PATH, so nothing about this kind can be verified here",
			"install " + bin + ", or set bin under [agents." + name + "] if it is named differently"}}
	}

	help, herr := helpText(ctx, path, timeout)
	if herr != nil {
		return []Finding{{name, "binary", Unknown, path + " did not produce help within the budget: " + herr.Error(),
			"run `" + bin + " --help` by hand; if it is interactive, set bin to a non-interactive wrapper"}}
	}
	if strings.TrimSpace(help) == "" {
		return []Finding{{name, "binary", Unknown, path + " printed no help output, so its flags cannot be read",
			"run `" + bin + " --help` by hand and set the flags under [agents." + name + "] to match"}}
	}

	findings := []Finding{{name, "binary", Accepted, path, ""}}
	lower := strings.ToLower(help)

	findings = append(findings, probeFlag(name, "model_flag", k.ModelFlag, lower, bin)...)
	if k.EffortFlag != "" {
		findings = append(findings, probeFlag(name, "effort_flag", k.EffortFlag, lower, bin)...)
	}

	// Model names. Found in the help is real evidence; absent is not evidence of
	// anything, so it is UNKNOWN and says why.
	for _, t := range policy.TierOrder {
		model := k.Tiers.Get(t)
		if model == "" {
			findings = append(findings, Finding{name, "tiers." + string(t), Rejected,
				"no model configured for the " + string(t) + " tier",
				"set tiers." + string(t) + " under [agents." + name + "]"})
			continue
		}
		if strings.Contains(lower, strings.ToLower(model)) {
			findings = append(findings, Finding{name, "tiers." + string(t), Accepted,
				model + " appears in " + bin + "'s own help", ""})
			continue
		}
		findings = append(findings, Finding{name, "tiers." + string(t), Unknown,
			model + " is not named in " + bin + "'s help, which usually lists flags but not models",
			"confirm `" + bin + " " + firstFlag(k) + " " + model + "` is accepted; if the vendor renamed it, set tiers." +
				string(t) + " under [agents." + name + "]"})
	}
	return findings
}

func probeFlag(kind, key, flag, lowerHelp, bin string) []Finding {
	if flag == "" {
		return nil
	}
	if strings.Contains(lowerHelp, strings.ToLower(flag)) {
		return []Finding{{kind, key, Accepted, bin + " documents " + flag, ""}}
	}
	// Help text exists and does not mention the flag. That IS evidence, and it is the
	// exact failure this probe was built for: a vendor renaming a flag leaves the
	// plugin logging confident decisions above arguments the agent ignores.
	return []Finding{{kind, key, Rejected,
		bin + " does not document " + flag + "; arguments built with it would be ignored or refused",
		"run `" + bin + " --help`, find the flag it uses now, and set " + key + " under [agents." + kind + "]"}}
}

func firstFlag(k Kind) string {
	if k.ModelFlag != "" {
		return k.ModelFlag
	}
	return "--model"
}

// helpText runs `<bin> --help` under a hard deadline.
//
// A non-zero exit is NOT treated as failure: several of these CLIs print help and exit
// 1, and discarding that output would turn every one of them into a false UNKNOWN.
// stdout and stderr are both captured for the same reason.
func helpText(ctx context.Context, path string, timeout time.Duration) (string, error) {
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, path, "--help")
	// Never inherit stdin: a binary that decides to prompt would otherwise hang until
	// the deadline, turning a fast check into a five-second stall per agent.
	cmd.Stdin = nil
	// WaitDelay is what actually makes the deadline real, and its absence is a trap a
	// test caught here: CommandContext kills the process on cancellation, but
	// CombinedOutput then blocks until the output pipes reach EOF — and a killed
	// shell's surviving grandchild still holds them open. Without this the probe ran
	// the full 30s of a `sleep 30` despite a 300ms budget. WaitDelay force-closes the
	// pipes shortly after the kill, so a wedged agent CLI cannot wedge `doctor`.
	cmd.WaitDelay = 250 * time.Millisecond
	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		return string(out), nil
	}
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	return "", err
}
