package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"text/tabwriter"
	"time"

	"github.com/muthuishere/herdr-jev/internal/agentkind"
	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/herdrapi"
	"github.com/muthuishere/herdr-jev/internal/jev"
)

// check is one prerequisite and, when it fails, what to do about it.
//
// A remediation is not decoration. This plugin's failures are QUIET by construction —
// it is designed never to block, so a completely dead backend looks exactly like a
// plugin that is simply not doing much. `doctor` is the only thing standing between
// that and a user who thinks it works. Every check that can fail names the exact
// command that fixes it.
type check struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
	Fix    string `json:"fix,omitempty"`
	// Unknown is the third state, and it exists because two are not enough: a
	// mapping that could not be verified is neither a pass nor a failure, and
	// collapsing it into either is how an unverifiable thing acquires a tick.
	// Last in the struct so the positional literals above stay readable.
	Unknown bool `json:"unknown,omitempty"`
}

// mark renders the three states distinctly. "??" is not a decoration: it is the whole
// point of having a third state at all.
func (c check) mark() string {
	switch {
	case c.Unknown:
		return "??  "
	case c.OK:
		return "ok  "
	}
	return "FAIL"
}

func doctor(flags map[string]string) error {
	var checks []check
	add := func(c check) { checks = append(checks, c) }

	// 1. Config parses. Everything downstream is read from it.
	cfg, cerr := config.Load()
	if cerr != nil {
		add(check{Name: "config", OK: false, Detail: cerr.Error(), Fix: "fix or delete " + config.Path()})
		return report(checks, flags)
	}
	add(check{Name: "config", OK: true, Detail: config.Path(), Fix: ""})

	// 2. The bars. Getting these inverted is silent and expensive: the plugin would
	// still run, still log, and spend money in the wrong direction all day.
	add(check{Name: "confidence bars", OK: true, Detail: fmt.Sprintf(
		"upgrade %.2f · downgrade %.2f · answer %.2f — spending more is the cheap mistake, so its bar is the lowest",
		cfg.Policy.MinUpgradeConfidence, cfg.Policy.MinDowngradeConfidence, cfg.Answer.MinConfidence), Fix: ""})

	// 3. The short circuit, and which mode it is in. "It never answers anything" is
	// almost always this, and it is the first thing anyone will ask about.
	switch {
	case cfg.Answer.Shadow:
		add(check{Name: "short circuit", OK: true, Detail: "SHADOW: logs what it would have answered, dispatches to an agent anyway", Fix: ""})
	case cfg.Answer.Enabled:
		add(check{Name: "short circuit", OK: true, Detail: fmt.Sprintf("ON, answering above %.2f confidence", cfg.Answer.MinConfidence), Fix: ""})
	default:
		add(check{Name: "short circuit", OK: false, Detail: "off (the default: answering instead of invoking your agent is opt-in)", Fix: "set answer.shadow = true in " + config.Path() + " to watch it for a week, then answer.enabled = true"})
	}

	// 4. Are we inside Herdr? Dispatch cannot start an agent otherwise.
	if os.Getenv("HERDR_ENV") == "" {
		add(check{Name: "herdr env", OK: false, Detail: "HERDR_ENV is not set; this is not a Herdr pane", Fix: "run inside a Herdr pane, or start one with `herdr`"})
	} else {
		add(check{Name: "herdr env", OK: true, Detail: "HERDR_ENV=1, pane " + firstNonEmpty(herdrapi.SelfPane(), "unknown"), Fix: ""})
	}

	// 5. Can we reach the Herdr server? `pane list` is the call dispatch makes.
	if panes, perr := herdrapi.New().Panes(ctx()); perr != nil {
		add(check{Name: "herdr server", OK: false, Detail: perr.Error(), Fix: "check HERDR_SOCKET_PATH; a named session has its own socket"})
	} else {
		add(check{Name: "herdr server", OK: true, Detail: fmt.Sprintf("%d pane(s)", len(panes)), Fix: ""})
	}

	// 6. openjev — the one dependency, and the one that degrades silently.
	client, berr := backend(cfg)
	if berr != nil {
		fix := "start it: openjev serve"
		if cfg.OpenJEV.Mode == "spawn" {
			// The un-downloaded-model case is the one that looks like a dead
			// server and is not. Consent for a first download is CLI-side and
			// needs a TTY (server ADRs 0006/0010); a supervised child has none, so
			// a first-ever spawn exits 77 having downloaded nothing. Reporting
			// that as "unreachable" sends someone debugging sockets.
			fix = "if openjev has never downloaded its weights here, a supervised child cannot ask for consent: run `" +
				cfg.OpenJEV.Bin + " model pull --yes` once (exit 77 means exactly this). Otherwise check openjev.bin in " +
				config.Path() + "; `" + cfg.OpenJEV.Bin + " --version` must work"
		}
		add(check{Name: "openjev", OK: false, Detail: berr.Error(), Fix: fix})
	} else {
		defer client.Close()
		add(check{Name: "openjev", OK: true, Detail: "reachable at " + client.URL, Fix: ""})

		// Ready and READY are different questions, and so are their fixes: a
		// download needs patience, a failure needs a log.
		if r, rerr := client.Ready(ctx()); rerr != nil {
			fix := "wait; this is a one-time cost"
			if r.Phase == "failed" {
				fix = "read the openjev server log; it is not retrying its way out of this"
			}
			add(check{Name: "openjev ready", OK: false, Detail: r.Human(), Fix: fix})
		} else {
			add(check{Name: "openjev ready", OK: true, Detail: r.Human(), Fix: ""})
		}

		if info, ierr := client.Info(ctx()); ierr != nil {
			add(check{Name: "openjev api", OK: false, Detail: ierr.Error(), Fix: "upgrade whichever of openjev or herdr-jev is older; this client speaks v" +
				fmt.Sprint(jev.APIVersion)})
		} else {
			model, device := "(not loaded yet)", "?"
			if info.Model != nil {
				model, device = info.Model.Model, info.Model.Device
			}
			add(check{Name: "openjev api", OK: true, Detail: fmt.Sprintf("v%d · %s on %s · max_options %d",
				info.APIVersion, model, device, info.Limits.MaxOptions), Fix: ""})

			// Which score key means entailment is registry config, so it is a fact
			// only the server has. Without it every probability this plugin reads is
			// zero — every answer refused, every classification discarded — and that
			// degradation is silent by design. It is worth its own check.
			if label := client.EntailmentLabel(ctx()); label != "" {
				add(check{Name: "entailment label", OK: true,
					Detail: "server reports " + label + "; scores are read by that key, never a guessed one", Fix: ""})
			} else {
				add(check{Name: "entailment label", OK: false,
					Detail: "the server has not reported one yet, so every probability would read as zero",
					Fix:    "wait for the model to finish loading; /v1/model is 503 model_not_ready until weights are resident"})
			}
		}
	}

	// 7. The agent-kind mappings, printed in full.
	//
	// A vendor renaming a flag produces an agent that is silently never routed — the
	// plugin logs a decision, builds arguments nothing accepts, and the model never
	// changes. Making the mapping visible is the only defence available to us.
	kinds := make([]string, 0, len(cfg.Agents))
	for k := range cfg.Agents {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	if len(kinds) == 0 {
		add(check{Name: "agent kinds", OK: false, Detail: "no [agents.*] sections configured", Fix: "add one, e.g. [agents.claude] with tiers.fast/balanced/deep"})
	}
	for _, name := range kinds {
		k := cfg.Agents[name]
		detail := fmt.Sprintf("%s → %s/%s/%s", firstNonEmpty(k.ModelFlag, k.ModelEnv, "NO MODEL CONTROL"),
			k.Tiers.Fast, k.Tiers.Balanced, k.Tiers.Deep)
		if !agentkind.Known(name) {
			add(check{Name: "agent " + name, OK: false, Detail: "herdr cannot start an agent of this kind", Fix: "rename the section to one of: " + fmt.Sprint(agentkind.HerdrKinds())})
			continue
		}
		if k.ModelFlag == "" && k.ModelEnv == "" {
			add(check{Name: "agent " + name, OK: false, Detail: "no model_flag and no model_env: nothing to route", Fix: "set model_flag (e.g. \"--model\") under [agents." + name + "]"})
			continue
		}
		if k.Tiers.Fast == "" || k.Tiers.Balanced == "" || k.Tiers.Deep == "" {
			add(check{Name: "agent " + name, OK: false, Detail: "a tier has no model name: " + detail, Fix: "fill in all three of tiers.fast / tiers.balanced / tiers.deep"})
			continue
		}
		add(check{Name: "agent " + name, OK: true, Detail: detail, Fix: ""})
	}

	// 8. --probe: opt-in, bounded verification that the mappings above are real.
	//
	// Plain `doctor` stays fast and offline. This is the only part that runs other
	// people's binaries, so it is never implied and never automatic — and it is a
	// CHECK, not a gate: routing works whether or not it has ever been run.
	if flags["probe"] != "" {
		checks = append(checks, probeKinds(cfg, kinds)...)
	} else if len(kinds) > 0 {
		add(check{Name: "mapping verified", OK: false, Detail: "the tier and flag mappings above are UNVERIFIED guesses against vendor CLIs", Fix: "herdr-jev doctor --probe — reads each agent's own --help and reports what it really accepts"})
	}

	return report(checks, flags)
}

// probeKinds runs the opt-in probe and folds its findings into the check list.
//
// UNKNOWN is deliberately NOT rendered as a pass. A mapping nobody could verify must
// read differently from a verified one, or the probe hands back the same false
// confidence it was built to remove — so it prints with its own marker and its own fix,
// and it does not count as a failure either, because "this agent is not installed here"
// is not something to fix.
func probeKinds(cfg config.Config, kinds []string) []check {
	timeout := 5 * time.Second
	var out []check
	for _, name := range kinds {
		for _, f := range agentkind.Probe(ctx(), name, cfg.Agents[name], timeout) {
			out = append(out, check{
				Name:    "probe " + f.Kind + " " + f.What,
				OK:      f.State != agentkind.Rejected,
				Detail:  f.Detail,
				Fix:     f.Fix,
				Unknown: f.State == agentkind.Unknown,
			})
		}
	}
	// Report only. An auto-fix that guessed a replacement model name would
	// reintroduce the same class of error with more confidence behind it.
	return out
}

func report(checks []check, flags map[string]string) error {
	failed, unknown := 0, 0
	for _, c := range checks {
		switch {
		case c.Unknown:
			unknown++
		case !c.OK:
			failed++
		}
	}
	if flags["json"] != "" {
		b, _ := json.MarshalIndent(map[string]any{
			"ok": failed == 0, "failed": failed, "unknown": unknown, "checks": checks}, "", "  ")
		fmt.Println(string(b))
	} else {
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		for _, c := range checks {
			fmt.Fprintf(w, "%s\t%s\t%s\n", c.mark(), c.Name, c.Detail)
			if (!c.OK || c.Unknown) && c.Fix != "" {
				fmt.Fprintf(w, "\t\t  -> %s\n", c.Fix)
			}
		}
		w.Flush()
		if unknown > 0 {
			fmt.Printf("\n?? = could not be verified here. Not a pass.\n")
		}
	}
	if failed > 0 {
		return fmt.Errorf("%d check(s) failed", failed)
	}
	return nil
}
