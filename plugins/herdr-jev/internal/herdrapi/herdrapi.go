// Package herdrapi is the only place that talks to Herdr.
//
// It shells out to the `herdr` binary rather than speaking the socket directly. That
// is a deliberate v1 choice: `herdr pane list|read` and `herdr agent list|prompt|wait`
// are stable, documented, already emit the JSON envelope, and enforce agent_blocked
// server-side before any input is sent. A hand-rolled socket client would be a second
// protocol implementation to keep correct for no behaviour we need.
package herdrapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Stable errors. Callers branch on these, never on message text.
var (
	ErrNotFound     = errors.New("pane not found")
	ErrBlocked      = errors.New("agent is blocked; a human is needed")
	ErrSelfDelivery = errors.New("refusing to route into our own pane")
	ErrTimeout      = errors.New("timed out")
	ErrNoHost       = errors.New("no herdr server reachable")
)

// Pane is one pane as `herdr pane list` reports it (PaneInfo, protocol 22). Only the
// fields the router actually distils are named; unknown fields are ignored so a Herdr
// upgrade that adds one does not break us.
type Pane struct {
	PaneID      string `json:"pane_id"`
	TerminalID  string `json:"terminal_id"`
	WorkspaceID string `json:"workspace_id"`
	TabID       string `json:"tab_id"`
	Focused     bool   `json:"focused"`
	Agent       string `json:"agent"`
	Status      string `json:"agent_status"`
	Cwd         string `json:"cwd"`
	ForegroundCwd string `json:"foreground_cwd"`
	Title       string `json:"terminal_title_stripped"`
	Label       string `json:"label"`
}

// Client runs herdr verbs.
type Client struct {
	Bin string
}

func New() *Client {
	bin := os.Getenv("HERDR_BIN_PATH")
	if bin == "" {
		bin = "herdr"
	}
	return &Client{Bin: bin}
}

// SelfPane is the pane we are running in, or "" outside Herdr.
//
// This is the only reason refusing to route to ourselves is possible at all: a router
// that does not know its own address cannot decline to wake itself, and an agent that
// wakes itself answers itself and routes again.
func SelfPane() string { return os.Getenv("HERDR_PANE_ID") }

type envelope struct {
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (c *Client) run(ctx context.Context, args ...string) (json.RawMessage, error) {
	out, err := exec.CommandContext(ctx, c.Bin, args...).Output()
	if len(out) == 0 && err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoHost, err)
	}
	var env envelope
	if jerr := json.Unmarshal(out, &env); jerr != nil {
		return nil, fmt.Errorf("herdr %s: unparseable output: %w", strings.Join(args, " "), jerr)
	}
	if env.Error != nil {
		switch env.Error.Code {
		case "pane_not_found", "agent_not_found":
			return nil, ErrNotFound
		case "agent_blocked":
			return nil, ErrBlocked
		case "timeout":
			return nil, ErrTimeout
		}
		return nil, fmt.Errorf("herdr %s: %s: %s", args[0], env.Error.Code, env.Error.Message)
	}
	return env.Result, nil
}

// Panes is the snapshot the whole router is built on: every pane in every workspace,
// as of right now.
//
// It MUST be re-run for every route. A pane can appear, close or change agent between
// one message and the next, and that is ordinary traffic, not an exception.
func (c *Client) Panes(ctx context.Context) ([]Pane, error) {
	raw, err := c.run(ctx, "pane", "list")
	if err != nil {
		return nil, err
	}
	var r struct {
		Panes []Pane `json:"panes"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, err
	}
	return r.Panes, nil
}

// Read returns a pane's recent output, ANSI stripped.
//
// `recent` rather than `visible`: a pane whose agent just printed a long answer has
// scrolled the interesting part off screen, and the interesting part is exactly what
// tells us what that agent is working on.
//
// A read failure is NOT fatal to routing. A pane we cannot read is still rankable on
// its identity line — degrading to less signal beats refusing to route at all.
func (c *Client) Read(ctx context.Context, paneID string, lines int) (string, error) {
	args := []string{"pane", "read", paneID, "--source", "recent"}
	if lines > 0 {
		args = append(args, "--lines", strconv.Itoa(lines))
	}
	raw, err := c.run(ctx, args...)
	if err != nil {
		return "", err
	}
	var r struct {
		Text string `json:"text"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return "", err
	}
	if r.Text != "" {
		return r.Text, nil
	}
	return r.Data, nil
}

// Resolve turns a handle into a LIVE pane.
//
// It accepts a pane id or a terminal id, because holds and journal entries record
// terminal_id — pane ids are recycled across server restarts, and a cached one
// silently resolves to somebody else's pane. Resolving fresh, every time, is what
// stops a held message reaching a stranger.
func (c *Client) Resolve(ctx context.Context, handle string) (Pane, error) {
	panes, err := c.Panes(ctx)
	if err != nil {
		return Pane{}, err
	}
	for _, p := range panes {
		if p.PaneID == handle || p.TerminalID == handle {
			return p, nil
		}
	}
	return Pane{}, fmt.Errorf("%w: %q", ErrNotFound, handle)
}

// Prompt delivers the message to a pane.
//
// It refuses self-delivery here rather than trusting every caller to remember. Herdr
// itself rejects a blocked target before any input is sent, so the dangerous case
// cannot slip past us — our job is to detect and report it, not to prevent it.
func (c *Client) Prompt(ctx context.Context, paneID, text string, wait bool, timeout time.Duration) error {
	if self := SelfPane(); self != "" && self == paneID {
		return ErrSelfDelivery
	}
	args := []string{"agent", "prompt", paneID, text}
	if wait {
		args = append(args, "--wait", "--until", "idle", "--until", "done", "--until", "blocked",
			"--timeout", strconv.FormatInt(timeout.Milliseconds(), 10))
	}
	_, err := c.run(ctx, args...)
	return err
}

// StartAgent starts an agent in an existing pane, with OUR chosen arguments.
//
// This is the single place the whole model-routing feature touches reality. Herdr's
// `agent.start` takes `{name, kind, pane_id, args[]}` and the pane must be sitting at
// an interactive shell prompt; the args are forwarded to the agent's own CLI after a
// `--`. Choosing them is choosing the model, and it is the only moment at which that
// choice is ours to make — Herdr offers nothing that changes a RUNNING agent's model.
func (c *Client) StartAgent(ctx context.Context, name, kind, paneID string, args []string) error {
	argv := []string{"agent", "start", name, "--kind", kind, "--pane", paneID}
	if len(args) > 0 {
		argv = append(argv, "--")
		argv = append(argv, args...)
	}
	_, err := c.run(ctx, argv...)
	return err
}
