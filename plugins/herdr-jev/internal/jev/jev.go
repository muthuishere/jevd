// Package jev is herdr-jev's entire relationship with inference: an HTTP client of a
// public `openjev serve` API, and nothing more.
//
// There is no model here, no tokenizer, no weights, and no scoring maths beyond
// comparing floats the server handed us. If the router needs something the API does
// not offer, the API gains it — for everyone. See SPEC.md §2 and the contract in
// docs/design/02-api-and-cli.md, which wins on any disagreement.
package jev

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// APIVersion is the major this client was built for. A mismatch is refused outright:
// a client guessing at v2 semantics is worse than a client that stops.
const APIVersion = 1

// DefaultPort is discovery step 4 only — the documented default, not a contract. The
// port is not the contract; the state file is.
const DefaultPort = 21131

var (
	ErrNoServer   = errors.New("no openjev server reachable")
	ErrNotReady   = errors.New("openjev is running but not ready")
	ErrAPIVersion = errors.New("openjev api version mismatch")
)

// Reranker is the seam every test in this repo runs against.
//
// One method, because the router needs exactly one thing from the model: given a
// question and N options, which options entail it and how strongly. Keeping it this
// narrow is what lets ranking, floor and margin be tested with no weights on disk.
type Reranker interface {
	Rerank(ctx context.Context, question string, options []string) ([]Scored, error)
}

// Scored is one option's result. Index refers to the caller's options slice — the
// caller already has the strings, so we never echo them back (return_documents stays
// false and a 200-option payload does not double).
type Scored struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

// APIError is the server's error envelope. `code` is the contract and `message` is
// not, so callers branch on Code and print Message.
type APIError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	Status    int    `json:"-"`
}

func (e *APIError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf("openjev %s: %s (request %s)", e.Code, e.Message, e.RequestID)
	}
	return fmt.Sprintf("openjev %s: %s", e.Code, e.Message)
}

// Info is /v1/info. Limits are READ, never hardcoded: max_options is what decides
// whether 600 panes need chunking, and discovering that ceiling with a 413 in front of
// a user is not discovery, it is a bug report.
type Info struct {
	APIVersion    int      `json:"api_version"`
	ServerVersion string   `json:"server_version"`
	Capabilities  []string `json:"capabilities"`
	Limits        struct {
		MaxOptions    int `json:"max_options"`
		MaxFieldChars int `json:"max_field_chars"`
	} `json:"limits"`
	Model struct {
		Ref      string `json:"ref"`
		Revision string `json:"revision"`
		Device   string `json:"device"`
	} `json:"model"`
}

// Readiness is /readyz. The phase split matters: "downloading 7.9 GB" and "loading
// onto metal" are different waits with different remedies, and a router that reports
// only "not ready" during a four-minute first-run download gets killed by an impatient
// human who assumes it is wedged.
type Readiness struct {
	Ready  bool   `json:"ready"`
	Phase  string `json:"phase"`
	Detail struct {
		File       string `json:"file"`
		BytesDone  int64  `json:"bytes_done"`
		BytesTotal int64  `json:"bytes_total"`
		ETASeconds int64  `json:"eta_seconds"`
	} `json:"detail"`
}

// Human renders a readiness state as one line a person can act on.
func (r Readiness) Human() string {
	switch r.Phase {
	case "downloading":
		s := "downloading model"
		if r.Detail.File != "" {
			s += " " + filepath.Base(r.Detail.File)
		}
		if r.Detail.BytesTotal > 0 {
			s += fmt.Sprintf(" %.1f/%.1f GB",
				float64(r.Detail.BytesDone)/1e9, float64(r.Detail.BytesTotal)/1e9)
		}
		if r.Detail.ETASeconds > 0 {
			s += fmt.Sprintf(", ETA %ds", r.Detail.ETASeconds)
		}
		return s + " (one time only)"
	case "loading":
		return "loading weights onto the device"
	case "ready":
		return "ready"
	case "failed":
		return "FAILED — see the openjev server log"
	case "":
		return "unknown"
	default:
		return r.Phase
	}
}

// Client is a discovered-or-spawned openjev server.
type Client struct {
	URL   string
	HTTP  *http.Client
	Token string

	// spawned, when non-nil, is the server WE started. Whoever spawned it kills it;
	// an attached server is never ours to signal.
	spawned *exec.Cmd
	limits  *Info
}

// Discover implements the ordered discovery of docs/design/02 §6.1, stopping at the
// first hit and confirming every hit with /healthz — a state file can outlive its
// process, and believing one blindly is how a router talks confidently to nothing.
func Discover(ctx context.Context, explicitURL string, timeout time.Duration) (*Client, error) {
	hc := &http.Client{Timeout: timeout}
	try := func(url string) *Client {
		if url == "" {
			return nil
		}
		c := &Client{URL: strings.TrimRight(url, "/"), HTTP: hc}
		if c.Healthy(ctx) {
			return c
		}
		return nil
	}

	if c := try(explicitURL); c != nil { // 1. explicit config
		return c, nil
	}
	if c := try(os.Getenv("OPENJEV_URL")); c != nil { // 2. environment
		return c, nil
	}
	for _, p := range statePaths() { // 3. state file(s)
		if u := urlFromStateFile(p); u != "" {
			if c := try(u); c != nil {
				return c, nil
			}
		}
	}
	if c := try(fmt.Sprintf("http://127.0.0.1:%d", DefaultPort)); c != nil { // 4. default
		return c, nil
	}
	return nil, ErrNoServer // 5. never guess further
}

// statePaths is our own spawned server's state file first, then the user's. Ours wins
// because if we spawned one, that is the one we own and the one we know the token for.
func statePaths() []string {
	var out []string
	if d := os.Getenv("HERDR_PLUGIN_STATE_DIR"); d != "" {
		out = append(out, filepath.Join(d, "openjev-server.json"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, filepath.Join(home, ".local", "state", "openjev", "server.json"))
	}
	return out
}

func urlFromStateFile(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var st struct {
		URL string `json:"url"`
	}
	if json.Unmarshal(b, &st) != nil {
		return ""
	}
	return st.URL
}

// Healthy is /healthz: "do not restart me". It is 200 while downloading and while
// loading, so it answers "is there a process there", never "can it serve".
func (c *Client) Healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.URL+"/healthz", nil)
	if err != nil {
		return false
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// Ready is /readyz: "send me traffic". Conflating it with /healthz is how a supervisor
// kills a process 40 seconds into a 90-second model load, forever.
func (c *Client) Ready(ctx context.Context) (Readiness, error) {
	var r Readiness
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.URL+"/readyz", nil)
	if err != nil {
		return r, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return r, fmt.Errorf("%w: %v", ErrNoServer, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	_ = json.Unmarshal(body, &r)
	if resp.StatusCode == http.StatusOK {
		r.Ready = true
		if r.Phase == "" {
			r.Phase = "ready"
		}
		return r, nil
	}
	return r, ErrNotReady
}

// Info fetches and caches /v1/info, refusing a foreign API major.
//
// Feature detection is via Capabilities, never version arithmetic: enable_latents=false
// is a capability difference at the SAME version, which no version comparison can
// express.
func (c *Client) Info(ctx context.Context) (Info, error) {
	if c.limits != nil {
		return *c.limits, nil
	}
	var info Info
	if err := c.getJSON(ctx, "/v1/info", &info); err != nil {
		return info, err
	}
	if info.APIVersion != APIVersion {
		return info, fmt.Errorf("%w: server speaks v%d, herdr-jev was built for v%d; upgrade whichever is older",
			ErrAPIVersion, info.APIVersion, APIVersion)
	}
	c.limits = &info
	return info, nil
}

// MaxOptions is the server's ceiling, with a conservative fallback when /v1/info is
// unavailable. Never a hardcoded 512.
func (c *Client) MaxOptions(ctx context.Context) int {
	info, err := c.Info(ctx)
	if err != nil || info.Limits.MaxOptions <= 0 {
		return 64
	}
	return info.Limits.MaxOptions
}

// MaxFieldChars is the per-field ceiling distillation truncates to.
func (c *Client) MaxFieldChars(ctx context.Context) int {
	info, err := c.Info(ctx)
	if err != nil || info.Limits.MaxFieldChars <= 0 {
		return 32768
	}
	return info.Limits.MaxFieldChars
}

// Rerank ranks options against the question, chunking to the server's max_options and
// merging by score.
//
// One call per chunk, not one per option: this is a cross-encoder that batches pairs
// internally, so N single-option calls would be N queue waits for one batch of work.
func (c *Client) Rerank(ctx context.Context, question string, options []string) ([]Scored, error) {
	if len(options) == 0 {
		return nil, nil
	}
	max := c.MaxOptions(ctx)
	var all []Scored
	for start := 0; start < len(options); start += max {
		end := min(start+max, len(options))
		chunk, err := c.rerankChunk(ctx, question, options[start:end])
		if err != nil {
			return nil, err
		}
		for _, s := range chunk {
			// Translate the chunk-local index back to the caller's, or a second
			// chunk silently overwrites the first's answers.
			s.Index += start
			all = append(all, s)
		}
	}
	return all, nil
}

func (c *Client) rerankChunk(ctx context.Context, question string, options []string) ([]Scored, error) {
	body, _ := json.Marshal(map[string]any{
		"question":         question,
		"options":          options,
		"return_scores":    true,
		"return_documents": false,
	})
	var out struct {
		Results []Scored `json:"results"`
	}
	if err := c.postJSON(ctx, "/v1/rerank", body, &out); err != nil {
		return nil, err
	}
	return out.Results, nil
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.URL+path, nil)
	if err != nil {
		return err
	}
	return c.do(req, out)
}

func (c *Client) postJSON(ctx context.Context, path string, body []byte, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(req, out)
}

func (c *Client) do(req *http.Request, out any) error {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNoServer, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		var env struct {
			Error APIError `json:"error"`
		}
		if json.Unmarshal(body, &env) == nil && env.Error.Code != "" {
			e := env.Error
			e.Status = resp.StatusCode
			return &e
		}
		return fmt.Errorf("openjev %s: HTTP %d", req.URL.Path, resp.StatusCode)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(body, out)
}

// Spawn starts our own server and blocks on its single ready line.
//
// --port 0 plus a PRIVATE state file is the whole trick: a spawned server can never
// collide with the user's own, and can never be found by somebody else's discovery, so
// killing ours cannot break theirs.
func Spawn(ctx context.Context, bin, model, stateFile string, timeout time.Duration) (*Client, error) {
	args := []string{"serve", "--port", "0", "--state-file", stateFile, "--print-ready-json"}
	if model != "" {
		args = append(args, "--model", model)
	}
	if err := os.MkdirAll(filepath.Dir(stateFile), 0o755); err != nil {
		return nil, err
	}
	cmd := exec.Command(bin, args...)
	cmd.Stderr = os.Stderr // logs are stderr; stdout carries exactly one JSON line
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("spawn %s: %w", bin, err)
	}

	type ready struct {
		URL        string `json:"url"`
		APIVersion int    `json:"api_version"`
	}
	lines := make(chan ready, 1)
	errs := make(chan error, 1)
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			var r ready
			if json.Unmarshal(sc.Bytes(), &r) == nil && r.URL != "" {
				lines <- r
				return
			}
		}
		errs <- errors.New("openjev exited before printing its ready line")
	}()

	// The wait is generous on purpose: a first run downloads several GB, and a
	// router that gives up at 30s turns a slow start into a broken install.
	select {
	case r := <-lines:
		if r.APIVersion != 0 && r.APIVersion != APIVersion {
			_ = cmd.Process.Kill()
			return nil, fmt.Errorf("%w: spawned server speaks v%d, we speak v%d", ErrAPIVersion, r.APIVersion, APIVersion)
		}
		return &Client{URL: strings.TrimRight(r.URL, "/"), HTTP: &http.Client{Timeout: timeout}, spawned: cmd}, nil
	case err := <-errs:
		return nil, err
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		return nil, ctx.Err()
	}
}

// Close stops a server WE spawned, and does nothing at all to one we merely attached
// to. Ownership is the rule: whoever spawned it kills it, nobody else signals it.
func (c *Client) Close() error {
	if c == nil || c.spawned == nil || c.spawned.Process == nil {
		return nil
	}
	_ = c.spawned.Process.Signal(os.Interrupt)
	done := make(chan error, 1)
	go func() { done <- c.spawned.Wait() }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = c.spawned.Process.Kill()
	}
	return nil
}

// Wait blocks until a spawned server exits. Only meaningful in spawn mode.
func (c *Client) Wait() error {
	if c == nil || c.spawned == nil {
		return nil
	}
	return c.spawned.Wait()
}

// Spawned reports whether this process owns the server.
func (c *Client) Spawned() bool { return c != nil && c.spawned != nil }

// Pair is one premise/hypothesis for /v1/predict.
type Pair struct {
	Premise    string `json:"premise"`
	Hypothesis string `json:"hypothesis"`
	ID         string `json:"id,omitempty"`
}

// Prediction is one pair's labelled result. `scores` is a keyed object, never a bare
// array, because an array's ordering becomes an undocumented tribal fact.
type Prediction struct {
	ID     string             `json:"id"`
	Index  int                `json:"index"`
	Label  string             `json:"label"`
	Scores map[string]float64 `json:"scores"`
}

// Entailment is P(entailment), the calibrated probability the whole policy reads as a
// confidence. Absent scores read as 0 rather than as a confident no.
func (p Prediction) Entailment() float64 { return p.Scores["entailment"] }

// Predict runs NLI over premise/hypothesis pairs.
//
// This is how a yes/no question gets a NUMBER instead of a sentence: the task is the
// premise, the claim is the hypothesis, and the entailment probability is the answer
// and its confidence at once. Results come back in request order.
func (c *Client) Predict(ctx context.Context, pairs []Pair) ([]Prediction, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	body, _ := json.Marshal(map[string]any{"pairs": pairs, "truncate": "tail"})
	var out struct {
		Results []Prediction `json:"results"`
	}
	if err := c.postJSON(ctx, "/v1/predict", body, &out); err != nil {
		return nil, err
	}
	return out.Results, nil
}

// Grade is answer-vs-reference. Used only for an explicitly graded task, never inferred.
type Grade struct {
	Label     string             `json:"label"`
	Scores    map[string]float64 `json:"scores"`
	Pass      bool               `json:"pass"`
	Threshold float64            `json:"threshold"`
}

func (c *Client) Grade(ctx context.Context, answer, reference string, threshold float64) (Grade, error) {
	var g Grade
	body, _ := json.Marshal(map[string]any{"answer": answer, "reference": reference, "threshold": threshold})
	err := c.postJSON(ctx, "/v1/grade", body, &g)
	return g, err
}
