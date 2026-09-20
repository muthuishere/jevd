// Package cli is the verb surface. The CLI, the plugin actions and the skill expose
// the same verbs; if one has a verb the others do not, that is a bug.
package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/muthuishere/herdr-jev/internal/answer"
	"github.com/muthuishere/herdr-jev/internal/config"
	"github.com/muthuishere/herdr-jev/internal/dispatch"
	"github.com/muthuishere/herdr-jev/internal/herdrapi"
	"github.com/muthuishere/herdr-jev/internal/jev"
	"github.com/muthuishere/herdr-jev/internal/policy"
	"github.com/muthuishere/herdr-jev/internal/route"
)

const usage = `herdr-jev — answer it, route it, or get out of the way.

  dispatch <task>     THE verb. Answer the task locally if it is decision-shaped and
                      the model is sure; otherwise pick the model tier and reasoning
                      effort and start an agent with them; otherwise pass it through.
    --kind <k>        agent kind (claude, codex, gemini, ...). Required.
    --pane <id>       existing pane to start the agent in (default: a new split)
    --model <m>       the model it would otherwise have used
    --effort <e>      the effort it would otherwise have used
    --option <text>   a closed option for a pick-one task; repeatable
    --reference <t>   grade the task text against this reference
    --dry-run         decide and explain; start nothing, answer nothing
    --explain         print the full ranking behind the decision

  ask <question>      short circuit only: answer locally or say why it declined
  classify <task>     print the raw tier/effort/risky answer, before any policy
  why [n]             the last n decisions, with what each one wanted and did
  savings             how many tasks never reached an agent
  status [--watch]    openjev phase and model, config in effect, savings
  feed                live decisions (pane entrypoint)
  doctor [--json] [--probe]
                      every check names its own fix. --probe additionally reads each
                      agent binary's own --help and reports what it really accepts.
  skill [--install]   print / symlink the agent skill
  version             the build this binary is
  daemon | serve      spawn-mode supervisor

  route <message>     SECONDARY: deliver a message to whichever OPEN pane it was for
  panes [--distilled] candidates as the pane router sees them
  hold list | resolve <id> --to <n|pane> | --drop

Every verb takes --json.`

// Build identity, set from main. See cmd/herdr-jev/main.go.
var (
	Version   = "dev"
	Commit    = "unknown"
	BuildDate = "unknown"
)

// Main is the entry point. It returns an exit code rather than calling os.Exit so the
// verbs stay testable.
func Main(argv []string) int {
	if len(argv) == 0 || argv[0] == "-h" || argv[0] == "--help" {
		fmt.Println(usage)
		return 0
	}
	verb, rest := argv[0], argv[1:]
	args, flags := parse(rest)

	var err error
	switch verb {
	case "dispatch":
		err = dispatchCmd(args, flags)
	case "ask":
		err = askCmd(args, flags)
	case "classify":
		err = classifyCmd(args, flags)
	case "why":
		err = whyCmd(args, flags)
	case "savings":
		err = savingsCmd(flags)
	case "status":
		err = statusCmd(flags)
	case "feed":
		err = feedCmd(flags)
	case "doctor":
		err = doctor(flags)
	case "skill":
		err = skill(flags)
	case "version", "--version", "-v":
		err = versionCmd(flags)
	case "daemon":
		err = daemonCmd()
	case "serve":
		err = serveCmd()
	case "route":
		err = routeCmd(args, flags)
	case "panes":
		err = panesCmd(flags)
	case "hold":
		err = holdCmd(args, flags)
	case "resolve":
		err = resolveCmd(args, flags)
	default:
		fmt.Fprintf(os.Stderr, "unknown verb %q\n\n%s\n", verb, usage)
		return 2
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "herdr-jev: "+err.Error())
		return exitCode(err)
	}
	return 0
}

// exitCode mirrors openjev's own code table where it overlaps, so a script wrapping
// both can branch on one set of numbers.
func exitCode(err error) int {
	switch {
	case errorsIs(err, jev.ErrNoServer):
		return 6
	case errorsIs(err, jev.ErrNotReady):
		return 7
	case errorsIs(err, jev.ErrAPIVersion):
		return 3
	}
	return 1
}

func errorsIs(err, target error) bool {
	for e := err; e != nil; {
		if e == target {
			return true
		}
		u, ok := e.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		e = u.Unwrap()
	}
	return false
}

// parse is a tiny flag reader: --flag value, --flag=value, and bare --flag. Repeatable
// flags accumulate under the same key, newline-separated.
func parse(argv []string) ([]string, map[string]string) {
	args := []string{}
	flags := map[string]string{}
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		if !strings.HasPrefix(a, "--") {
			args = append(args, a)
			continue
		}
		name := strings.TrimPrefix(a, "--")
		value := "1"
		if j := strings.Index(name, "="); j >= 0 {
			name, value = name[:j], name[j+1:]
		} else if i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "--") && takesValue(name) {
			i++
			value = argv[i]
		}
		if prev, ok := flags[name]; ok && repeatable(name) {
			flags[name] = prev + "\n" + value
		} else {
			flags[name] = value
		}
	}
	return args, flags
}

func takesValue(name string) bool {
	switch name {
	case "json", "dry-run", "explain", "watch", "install", "distilled", "force", "wait", "drop", "shadow", "probe":
		return false
	}
	return true
}

func repeatable(name string) bool { return name == "option" }

func multi(flags map[string]string, name string) []string {
	v, ok := flags[name]
	if !ok || v == "" {
		return nil
	}
	return strings.Split(v, "\n")
}

func ctx() context.Context { return context.Background() }

// backend discovers or spawns openjev.
//
// A nil client is NOT an error here. Every caller degrades: dispatch passes the task
// through, doctor reports it with a fix. The plugin failing closed would mean an unwell
// classifier could stop a user's work, which is the one thing it must never do.
func backend(cfg config.Config) (*jev.Client, error) {
	timeout, err := cfg.HTTPTimeout()
	if err != nil {
		return nil, err
	}
	if cfg.OpenJEV.Mode == "spawn" {
		return jev.Spawn(ctx(), cfg.OpenJEV.Bin, cfg.OpenJEV.Model, config.ServerStatePath(), timeout)
	}
	return jev.Discover(ctx(), cfg.OpenJEV.URL, timeout)
}

// --- dispatch ----------------------------------------------------------------

func dispatchCmd(args []string, flags map[string]string) error {
	if len(args) == 0 {
		return fmt.Errorf("dispatch needs a task: herdr-jev dispatch \"...\" --kind claude")
	}
	task := strings.Join(args, " ")
	kind := flags["kind"]
	if kind == "" {
		return fmt.Errorf("dispatch needs --kind (one of: %s)", strings.Join(agentKindList(), ", "))
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	client, berr := backend(cfg)
	var b dispatch.Backend
	if berr == nil && client != nil {
		b = client
		defer client.Close()
	}

	req := dispatch.Request{
		Task:           task,
		Kind:           kind,
		Options:        multi(flags, "option"),
		GradeReference: flags["reference"],
		Current:        policy.Current{Model: flags["model"], Effort: currentEffort(flags["effort"])},
	}
	plan := dispatch.Dispatcher{Cfg: cfg, Backend: b}.Decide(ctx(), req)
	if berr != nil {
		plan.Lines = append([]string{"openjev unavailable: " + berr.Error()}, plan.Lines...)
	}

	dry := flags["dry-run"] != ""
	rec := dispatch.Record{V: 1, TS: time.Now().UTC(), Task: task, Plan: plan, DryRun: dry}

	// Execute, unless this is a dry run.
	if !dry && plan.Stage != dispatch.StageAnswered {
		if xerr := execute(cfg, plan, req, flags); xerr != nil {
			rec.Error = xerr.Error()
			_ = dispatch.Append(config.JournalPath(), rec, cfg.Journal.Keep)
			return xerr
		}
	}
	_ = dispatch.Append(config.JournalPath(), rec, cfg.Journal.Keep)

	if flags["json"] != "" {
		return printJSON(plan)
	}
	for _, l := range plan.Lines {
		fmt.Println("[herdr-jev] " + l)
	}
	if plan.Stage == dispatch.StageAnswered && !dry {
		fmt.Println()
		fmt.Println(plan.Answer.Text)
	}
	if flags["explain"] != "" {
		printExplain(plan)
	}
	return nil
}

// execute starts the agent with the chosen arguments and hands it the task.
//
// This is the ONLY place the plan meets Herdr, and what it can do is bounded by what
// Herdr offers: a pane can be split with an environment, an agent can be started with
// arguments, and a running agent can be prompted. A running agent's model cannot be
// changed by anyone — see agentkind's package comment.
func execute(cfg config.Config, plan dispatch.Plan, req dispatch.Request, flags map[string]string) error {
	h := herdrapi.New()
	pane := flags["pane"]
	if pane == "" {
		return fmt.Errorf("no --pane given: starting a fresh pane is not wired up in this build, so name an existing pane at a shell prompt")
	}
	name := flags["name"]
	if name == "" {
		name = req.Kind
	}
	if err := h.StartAgent(ctx(), name, req.Kind, pane, plan.Launch.Args); err != nil {
		return err
	}
	timeout, _ := cfg.HTTPTimeout()
	return h.Prompt(ctx(), pane, req.Task, flags["wait"] != "", timeout)
}

func currentEffort(s string) any {
	if s == "" {
		return nil
	}
	// A numeric effort is the caller's own scale and the policy leaves it alone, so
	// it has to survive as a number rather than becoming the string "4000".
	if n, err := strconv.ParseFloat(s, 64); err == nil {
		return n
	}
	return s
}

func agentKindList() []string {
	cfg := config.Default()
	out := make([]string, 0, len(cfg.Agents))
	for k := range cfg.Agents {
		out = append(out, k)
	}
	return out
}

// --- ask / classify ----------------------------------------------------------

func askCmd(args []string, flags map[string]string) error {
	if len(args) == 0 {
		return fmt.Errorf("ask needs a question")
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	client, err := backend(cfg)
	if err != nil {
		return err
	}
	defer client.Close()

	res := answer.Try(ctx(), client, answer.Request{
		Task:           strings.Join(args, " "),
		Options:        multi(flags, "option"),
		GradeReference: flags["reference"],
		GradeThreshold: cfg.Answer.GradeThreshold,
		MinConfidence:  cfg.Answer.MinConfidence,
	})
	if flags["json"] != "" {
		return printJSON(res)
	}
	fmt.Println("[herdr-jev] " + res.Describe())
	if res.Answer != nil {
		fmt.Println()
		for _, o := range res.Answer.Options {
			fmt.Printf("  %.3f  %s\n", o.Probability, o.Text)
		}
		if res.Verdict == answer.VerdictAnswered {
			fmt.Printf("\n%s\n", res.Answer.Text)
		}
	}
	return nil
}

func classifyCmd(args []string, flags map[string]string) error {
	if len(args) == 0 {
		return fmt.Errorf("classify needs a task")
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	client, err := backend(cfg)
	if err != nil {
		return err
	}
	defer client.Close()

	plan := dispatch.Dispatcher{Cfg: cfg, Backend: client}.Decide(ctx(), dispatch.Request{
		Task: strings.Join(args, " "),
		Kind: firstNonEmpty(flags["kind"], "claude"),
	})
	if flags["json"] != "" {
		return printJSON(plan)
	}
	for _, l := range plan.Lines {
		fmt.Println("[herdr-jev] " + l)
	}
	return nil
}

// --- why / savings / status / feed --------------------------------------------

func whyCmd(args []string, flags map[string]string) error {
	limit := 1
	if len(args) > 0 {
		if n, err := strconv.Atoi(args[0]); err == nil {
			limit = n
		}
	}
	records, err := dispatch.Read(config.JournalPath(), limit)
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return fmt.Errorf("no decisions journalled yet at %s", config.JournalPath())
	}
	if flags["json"] != "" {
		return printJSON(records)
	}
	for _, r := range records {
		fmt.Printf("%s  %s\n  task: %s\n", r.TS.Local().Format(time.RFC3339), r.Plan.Stage, r.Task)
		for _, l := range r.Plan.Lines {
			fmt.Println("  " + l)
		}
		printExplain(r.Plan)
		fmt.Println()
	}
	return nil
}

func printExplain(p dispatch.Plan) {
	if p.ShortCircuit != nil {
		fmt.Printf("  shape: %s — %s\n", p.ShortCircuit.Detection.Shape, p.ShortCircuit.Detection.Reason)
		if a := p.ShortCircuit.Answer; a != nil {
			for _, o := range a.Options {
				fmt.Printf("    %.3f  %s\n", o.Probability, o.Text)
			}
		}
	}
	if len(p.Launch.Args) > 0 {
		fmt.Printf("  args: %s\n", strings.Join(p.Launch.Args, " "))
	}
	for k, v := range p.Launch.Env {
		fmt.Printf("  env:  %s=%s\n", k, v)
	}
}

func savingsCmd(flags map[string]string) error {
	records, err := dispatch.Read(config.JournalPath(), 0)
	if err != nil {
		return err
	}
	s := dispatch.Count(records)
	if flags["json"] != "" {
		return printJSON(s)
	}
	fmt.Printf("%d task(s) dispatched\n", s.Total)
	fmt.Printf("  %d answered locally  — agent calls avoided, 0 tokens, nothing left the machine\n", s.Answered)
	fmt.Printf("  %d routed            — model or effort chosen for them\n", s.Routed)
	fmt.Printf("  %d passed through    — left exactly as built\n", s.Passed)
	if s.ShadowWould > 0 {
		fmt.Printf("  %d WOULD have been answered locally (shadow mode; turn answer.enabled on to collect them)\n", s.ShadowWould)
	}
	return nil
}

func statusCmd(flags map[string]string) error {
	once := func() error {
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		out := map[string]any{
			"mode":           cfg.OpenJEV.Mode,
			"answer_enabled": cfg.Answer.Enabled,
			"answer_shadow":  cfg.Answer.Shadow,
			"answer_bar":     cfg.Answer.MinConfidence,
			"upgrade_bar":    cfg.Policy.MinUpgradeConfidence,
			"downgrade_bar":  cfg.Policy.MinDowngradeConfidence,
		}
		client, berr := backend(cfg)
		if berr != nil {
			out["openjev"] = "unreachable: " + berr.Error()
		} else {
			defer client.Close()
			out["url"] = client.URL
			r, _ := client.Ready(ctx())
			out["phase"] = r.Human()
			if info, ierr := client.Info(ctx()); ierr == nil && info.Model != nil {
				out["model"] = info.Model.Model
				out["device"] = info.Model.Device
				out["entailment_label"] = info.Model.EntailmentLabel
			}
		}
		records, _ := dispatch.Read(config.JournalPath(), 0)
		out["savings"] = dispatch.Count(records)

		if flags["json"] != "" {
			return printJSON(out)
		}
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		for _, k := range []string{"mode", "url", "phase", "model", "device", "entailment_label", "openjev",
			"answer_enabled", "answer_shadow", "answer_bar", "upgrade_bar", "downgrade_bar"} {
			if v, ok := out[k]; ok {
				fmt.Fprintf(w, "%s\t%v\n", k, v)
			}
		}
		w.Flush()
		s := out["savings"].(dispatch.Savings)
		fmt.Printf("\n%d dispatched · %d answered locally · %d routed · %d passed through\n",
			s.Total, s.Answered, s.Routed, s.Passed)
		return nil
	}
	if flags["watch"] == "" {
		return once()
	}
	for {
		fmt.Print("\033[H\033[2J")
		if err := once(); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		time.Sleep(3 * time.Second)
	}
}

// feedCmd tails the journal. This is the pane entrypoint: routing decisions visible
// while you work, not only afterwards.
func feedCmd(flags map[string]string) error {
	seen := 0
	for {
		records, err := dispatch.Read(config.JournalPath(), 0)
		if err == nil && len(records) > seen {
			// Read returns newest first; print the new ones oldest first.
			fresh := records[:len(records)-seen]
			for i := len(fresh) - 1; i >= 0; i-- {
				r := fresh[i]
				fmt.Printf("%s  %s  %s\n", r.TS.Local().Format("15:04:05"), r.Plan.Stage, truncate(r.Task, 60))
				for _, l := range r.Plan.Lines {
					fmt.Println("    " + l)
				}
			}
			seen = len(records)
		}
		time.Sleep(time.Second)
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

// --- pane routing (secondary) --------------------------------------------------

func routeCmd(args []string, flags map[string]string) error {
	if len(args) == 0 {
		return fmt.Errorf("route needs a message")
	}
	message := strings.Join(args, " ")
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	h := herdrapi.New()
	timeout, _ := cfg.HTTPTimeout()

	if to := flags["to"]; to != "" {
		p, rerr := h.Resolve(ctx(), to)
		if rerr != nil {
			return rerr
		}
		return h.Prompt(ctx(), p.PaneID, message, flags["wait"] != "", timeout)
	}

	client, err := backend(cfg)
	if err != nil {
		return err
	}
	defer client.Close()

	r := route.Router{Herdr: h, Reranker: client, Cfg: cfg, MaxField: client.MaxFieldChars(ctx())}
	cands, skipped, err := r.Snapshot(ctx())
	if err != nil {
		return err
	}
	d, err := r.Rank(ctx(), message, cands)
	if err != nil {
		return err
	}
	if flags["force"] != "" {
		d = route.Force(d)
	}

	entry := route.Entry{V: 1, ID: route.NewID("d"), TS: time.Now().UTC(), Message: message,
		Outcome: d.Outcome, Reason: d.Reason, Floor: d.Floor, Margin: d.Margin,
		Ranked: d.Ranked, Excluded: skipped, DryRun: flags["dry-run"] != ""}

	target := d.Target()
	if target != nil && flags["dry-run"] == "" {
		if perr := h.Prompt(ctx(), target.PaneID, message, flags["wait"] != "", timeout); perr != nil {
			entry.Error = perr.Error()
			_ = route.AppendJournal(config.JournalPath()+".panes", entry, cfg.Journal.Keep)
			return perr
		}
		entry.Target = &route.Target{PaneID: target.PaneID, TerminalID: target.TerminalID, Agent: target.Agent}
	}
	if d.Outcome == route.OutcomeHeld && flags["dry-run"] == "" {
		// Held, not dropped: the message survives with its ranking so resolving it
		// is picking from a list rather than retyping it.
		_ = route.AppendHold(config.HoldsPath(), route.Hold{
			ID: route.NewID("h"), TS: time.Now().UTC(), Message: message, Reason: d.Reason, Ranked: d.Ranked})
	}
	_ = route.AppendJournal(config.JournalPath()+".panes", entry, cfg.Journal.Keep)

	if flags["json"] != "" {
		return printJSON(entry)
	}
	fmt.Printf("%s: %s\n", d.Outcome, d.Reason)
	for _, r := range d.Ranked {
		fmt.Printf("  %d  %.3f  %-8s %-24s %s\n", r.Rank+1, r.Score, r.Agent, r.Cwd, truncate(r.Title, 40))
	}
	return nil
}

func panesCmd(flags map[string]string) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	r := route.Router{Herdr: herdrapi.New(), Cfg: cfg, MaxField: 32768}
	cands, skipped, err := r.Snapshot(ctx())
	if err != nil {
		return err
	}
	if flags["json"] != "" {
		return printJSON(map[string]any{"candidates": cands, "excluded": skipped})
	}
	for _, c := range cands {
		fmt.Printf("%-10s %-8s %-24s %s\n", c.PaneID, c.Agent, c.Cwd, truncate(c.Title, 40))
		if flags["distilled"] != "" {
			fmt.Printf("  ---- exactly what the model sees (%s) ----\n  %s\n",
				c.OptionSHA, strings.ReplaceAll(c.Option, "\n", "\n  "))
		}
	}
	for _, s := range skipped {
		fmt.Printf("%-10s SKIPPED  %s\n", s.PaneID, s.Reason)
	}
	return nil
}

func holdCmd(args []string, flags map[string]string) error {
	holds, err := route.Holds(config.HoldsPath())
	if err != nil {
		return err
	}
	if flags["json"] != "" {
		return printJSON(holds)
	}
	if len(holds) == 0 {
		fmt.Println("no held messages")
		return nil
	}
	for _, h := range holds {
		fmt.Printf("%s  %s\n  %s\n", h.ID, h.Reason, h.Message)
		for _, r := range h.Ranked {
			fmt.Printf("    %d  %.3f  %-8s %s\n", r.Rank+1, r.Score, r.Agent, r.Cwd)
		}
	}
	return nil
}

func resolveCmd(args []string, flags map[string]string) error {
	if len(args) == 0 {
		return fmt.Errorf("resolve needs a hold id")
	}
	id := args[0]
	holds, err := route.Holds(config.HoldsPath())
	if err != nil {
		return err
	}
	var hold *route.Hold
	for i := range holds {
		if holds[i].ID == id {
			hold = &holds[i]
		}
	}
	if hold == nil {
		return fmt.Errorf("no held message with id %q", id)
	}
	if flags["drop"] != "" {
		return route.RemoveHold(config.HoldsPath(), id)
	}
	to := flags["to"]
	if to == "" {
		return fmt.Errorf("resolve needs --to <n|pane-id|terminal-id>, or --drop")
	}
	// A bare number is a position in the ranking the hold carries, which is why the
	// ranking is stored with it: picking from a list beats retyping a pane id.
	if n, cerr := strconv.Atoi(to); cerr == nil {
		if n < 1 || n > len(hold.Ranked) {
			return fmt.Errorf("--to %d is outside the ranking (1..%d)", n, len(hold.Ranked))
		}
		to = hold.Ranked[n-1].TerminalID
	}
	h := herdrapi.New()
	p, rerr := h.Resolve(ctx(), to)
	if rerr != nil {
		return rerr
	}
	cfg, _ := config.Load()
	timeout, _ := cfg.HTTPTimeout()
	if perr := h.Prompt(ctx(), p.PaneID, hold.Message, false, timeout); perr != nil {
		return perr
	}
	// Removed only AFTER delivery succeeded. Removing first would lose the message,
	// and a router that eats messages is worse than one that misroutes them.
	return route.RemoveHold(config.HoldsPath(), id)
}

// --- plumbing ----------------------------------------------------------------

func printJSON(v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(b))
	return nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
