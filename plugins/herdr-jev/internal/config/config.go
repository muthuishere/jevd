// Package config is the plugin's TOML config and its state-directory layout.
//
// Config lives OUTSIDE the repo, in $HERDR_PLUGIN_CONFIG_DIR, because a checkout is
// something you `git pull`, and a pull that silently changes where messages get
// delivered is the worst kind of surprise.
package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/muthuishere/herdr-jev/internal/agentkind"
	"github.com/muthuishere/herdr-jev/internal/policy"
	"github.com/pelletier/go-toml/v2"
)

// Config is the whole file. Every field has a default that works with no file at all:
// a router that cannot start until it is configured is a router nobody ever tries.
type Config struct {
	OpenJEV OpenJEV                   `toml:"openjev"`
	Answer  Answer                    `toml:"answer"`
	Policy  Policy                    `toml:"policy"`
	Agents  map[string]agentkind.Kind `toml:"agents"`
	Routing Routing                   `toml:"routing"`
	Journal Journal                   `toml:"journal"`
}

// Answer configures the SHORT CIRCUIT — the headline feature, and the one that changes
// behaviour most, so it is the one with the most conservative defaults.
type Answer struct {
	// Enabled is FALSE by default, on purpose. Answering a task instead of invoking
	// the user's agent is a large behavioural claim to make on their behalf, and a
	// plugin that starts doing it the moment it is installed has made that claim
	// without being asked. It is opt-in, and it says so loudly the first time it
	// engages.
	Enabled bool `toml:"enabled"`
	// Shadow logs what the short circuit WOULD have done while dispatching to the
	// agent anyway. This is how the bar earns trust: run shadow for a week, read the
	// journal, then turn Enabled on with evidence instead of hope.
	Shadow bool `toml:"shadow"`
	// MinConfidence is the answer bar, SEPARATE from and HIGHER than the routing
	// bars. 0.85 because the two decisions are not comparable: the routing bars
	// trade money against capability and a wrong call is recoverable within the same
	// turn, while a wrong answer here REPLACES the work — the user gets a two-letter
	// reply where an agent would have looked at the code. At 0.85 a normalised
	// two-way answer must be roughly 6:1, which no genuine coin flip reaches, and a
	// four-way pick must beat a uniform 0.25 by a factor of three.
	MinConfidence float64 `toml:"min_confidence"`
	// GradeThreshold is the entailment level at which a graded answer passes.
	GradeThreshold float64 `toml:"grade_threshold"`
}

// Policy is the asymmetric-confidence machinery of policy.Config, in config form.
type Policy struct {
	// MinUpgradeConfidence: the bar to spend MORE. Low, because being wrong costs
	// money and money is recoverable.
	MinUpgradeConfidence float64 `toml:"min_upgrade_confidence"`
	// MinDowngradeConfidence: the bar to spend LESS. High, because being wrong means
	// a real task handled by too small a model, and that is not recoverable by
	// noticing the bill.
	MinDowngradeConfidence float64 `toml:"min_downgrade_confidence"`
	// BudgetMS is the latency budget for a whole classification. Past it the task is
	// dispatched exactly as built.
	BudgetMS int `toml:"budget_ms"`
	// RouteModel / RouteEffort are the two switches. Both on: unlike the Claude Code
	// mod this is ported from, we choose these at process START, so there is no
	// prompt cache to invalidate and no reason to keep the model switch off.
	RouteModel  bool `toml:"route_model"`
	RouteEffort bool `toml:"route_effort"`
	// LogDecisions writes the two-line transcript record for every decision.
	LogDecisions bool `toml:"log_decisions"`
}

// PolicyFor builds the policy config for one agent kind. Per-kind, because the tier
// names and the family hints are per-kind and a shared one would be wrong for all but
// the first agent anyone used.
func (c Config) PolicyFor(kind string) (policy.Config, agentkind.Kind, error) {
	k, ok := c.Agents[kind]
	if !ok {
		return policy.Config{}, agentkind.Kind{}, fmt.Errorf("no tier mapping for agent kind %q; add an [agents.%s] section to %s", kind, kind, Path())
	}
	return policy.Config{
		Tiers:                  k.Tiers,
		MinUpgradeConfidence:   c.Policy.MinUpgradeConfidence,
		MinDowngradeConfidence: c.Policy.MinDowngradeConfidence,
		FamilyHints:            k.FamilyHints,
	}, k, nil
}

// OpenJEV selects between the two modes of SPEC.md §2. Both are first-class.
type OpenJEV struct {
	// Mode is "attach" (discover a server, never own it) or "spawn" (run our own
	// on --port 0 with a private state file, and kill it when we exit).
	Mode string `toml:"mode"`
	// URL is discovery step 1 and wins over every later step when set.
	URL     string `toml:"url"`
	Timeout string `toml:"timeout"`
	// Bin and Model are spawn-mode only.
	Bin   string `toml:"bin"`
	Model string `toml:"model"`
}

// Routing holds the two numbers the whole design turns on.
type Routing struct {
	// Floor: below this, HOLD. Routing confidently to the wrong agent is the
	// failure this plugin exists to avoid, and a floor is how "I don't know" is
	// expressible at all.
	Floor float64 `toml:"floor"`
	// Margin: top minus second must clear this. A floor alone is not enough —
	// 0.86 vs 0.85 clears any sane floor and is still a coin flip.
	Margin float64 `toml:"margin"`

	MaxOutputChars int  `toml:"max_output_chars"`
	OutputLines    int  `toml:"output_lines"`
	IncludeWorking bool `toml:"include_working"`

	ExcludeAgents     []string `toml:"exclude_agents"`
	ExcludeWorkspaces []string `toml:"exclude_workspaces"`
}

type Journal struct {
	Keep int `toml:"keep"`
}

// Default is the config you get with no file. The numbers are stated in SPEC.md §4
// and §6; if they disagree, SPEC.md is wrong or this is, and both are a bug.
func Default() Config {
	return Config{
		OpenJEV: OpenJEV{Mode: "attach", Timeout: "20s", Bin: "openjev"},
		Answer: Answer{
			Enabled:        false,
			Shadow:         false,
			MinConfidence:  0.85,
			GradeThreshold: 0.5,
		},
		Policy: Policy{
			MinUpgradeConfidence:   0.30,
			MinDowngradeConfidence: 0.60,
			BudgetMS:               800,
			RouteModel:             true,
			RouteEffort:            true,
			LogDecisions:           true,
		},
		Agents: agentkind.Defaults(),
		Routing: Routing{
			Floor:          0.55,
			Margin:         0.10,
			MaxOutputChars: 1200,
			OutputLines:    40,
			IncludeWorking: true,
		},
		Journal: Journal{Keep: 500},
	}
}

// Dir is $HERDR_PLUGIN_CONFIG_DIR, or an XDG-ish fallback so the CLI works from a
// plain checkout outside Herdr.
func Dir() string {
	if d := os.Getenv("HERDR_PLUGIN_CONFIG_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "herdr-jev")
}

// StateDir holds the decision journal, the holds, and the spawn-mode pidfile. It is
// generated data, never config, and is safe to delete.
func StateDir() string {
	if d := os.Getenv("HERDR_PLUGIN_STATE_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "state", "herdr-jev")
}

func Path() string        { return filepath.Join(Dir(), "config.toml") }
func JournalPath() string { return filepath.Join(StateDir(), "decisions.ndjson") }
func HoldsPath() string   { return filepath.Join(StateDir(), "holds.ndjson") }
func PidPath() string     { return filepath.Join(StateDir(), "herdr-jev.pid") }

// ServerStatePath is the state file a SPAWNED openjev writes. It is deliberately ours
// and not ~/.local/state/openjev/server.json: a spawned server must never be found by
// somebody else's discovery, or stopping ours breaks theirs.
func ServerStatePath() string { return filepath.Join(StateDir(), "openjev-server.json") }

// Load reads the config, applying defaults for anything absent. A missing file is not
// an error — it is the common case.
func Load() (Config, error) {
	cfg := Default()
	b, err := os.ReadFile(Path())
	if errors.Is(err, os.ErrNotExist) {
		return cfg, nil
	}
	if err != nil {
		return cfg, err
	}
	if err := toml.Unmarshal(b, &cfg); err != nil {
		return cfg, fmt.Errorf("%s: %w", Path(), err)
	}
	return cfg, cfg.Validate()
}

// Validate rejects a config that would route badly rather than letting it route badly.
//
// A floor of 0 means "always deliver, however weak the match", which is exactly the
// behaviour this plugin exists to prevent — so it is refused, loudly, at load, not
// discovered later from a journal full of misroutes.
func (c Config) Validate() error {
	switch c.OpenJEV.Mode {
	case "attach", "spawn":
	default:
		return fmt.Errorf("openjev.mode must be \"attach\" or \"spawn\", got %q", c.OpenJEV.Mode)
	}
	if c.Routing.Floor <= 0 || c.Routing.Floor >= 1 {
		return fmt.Errorf("routing.floor must be in (0,1), got %v; 0 would deliver every message to whatever scored highest, however badly", c.Routing.Floor)
	}
	if c.Routing.Margin < 0 || c.Routing.Margin >= 1 {
		return fmt.Errorf("routing.margin must be in [0,1), got %v", c.Routing.Margin)
	}
	if c.Routing.MaxOutputChars < 0 || c.Routing.OutputLines < 0 {
		return errors.New("routing.max_output_chars and routing.output_lines must not be negative")
	}
	if c.Answer.MinConfidence <= 0 || c.Answer.MinConfidence >= 1 {
		return fmt.Errorf("answer.min_confidence must be in (0,1), got %v", c.Answer.MinConfidence)
	}
	if c.Answer.MinConfidence < c.Policy.MinDowngradeConfidence {
		// Not a style rule. The answer bar decides whether a human gets a machine's
		// two-word reply instead of their agent's work; if it were the LOWER of the
		// two, the plugin would be more willing to replace the task than to make it
		// cheaper, which is exactly backwards.
		return fmt.Errorf("answer.min_confidence (%v) must not be below policy.min_downgrade_confidence (%v): answering instead of working is a bigger claim than spending less",
			c.Answer.MinConfidence, c.Policy.MinDowngradeConfidence)
	}
	if c.Policy.MinUpgradeConfidence <= 0 || c.Policy.MinUpgradeConfidence >= 1 ||
		c.Policy.MinDowngradeConfidence <= 0 || c.Policy.MinDowngradeConfidence >= 1 {
		return errors.New("policy.min_upgrade_confidence and policy.min_downgrade_confidence must be in (0,1)")
	}
	if c.Policy.MinUpgradeConfidence > c.Policy.MinDowngradeConfidence {
		return fmt.Errorf("policy.min_upgrade_confidence (%v) is above min_downgrade_confidence (%v); that inverts the whole design — spending more is the cheap mistake and must have the LOWER bar",
			c.Policy.MinUpgradeConfidence, c.Policy.MinDowngradeConfidence)
	}
	if _, err := c.HTTPTimeout(); err != nil {
		return fmt.Errorf("openjev.timeout: %w", err)
	}
	return nil
}

// Budget is the classification latency budget.
func (c Config) Budget() time.Duration {
	if c.Policy.BudgetMS <= 0 {
		return 800 * time.Millisecond
	}
	return time.Duration(c.Policy.BudgetMS) * time.Millisecond
}

// HTTPTimeout parses openjev.timeout. Generous by default: a first-run model load is
// tens of seconds and timing out on it produces a "broken" router that is merely warm.
func (c Config) HTTPTimeout() (time.Duration, error) {
	if c.OpenJEV.Timeout == "" {
		return 20 * time.Second, nil
	}
	return time.ParseDuration(c.OpenJEV.Timeout)
}
