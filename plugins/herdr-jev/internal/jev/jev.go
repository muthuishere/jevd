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
	"strconv"
	"strings"
	"time"
)

// APIVersion is the major this client was built for. A mismatch is refused outright:
// a client guessing at v2 semantics is worse than a client that stops.
const APIVersion = 1

// ExitConsentRequired is openjev's exit code for "a download needs permission" (server
// ADR 0010). It sits deliberately outside design 02's 0-9 table, so it is named here
// rather than compared as a bare number at a call site.
const ExitConsentRequired = 77

// DefaultPort is discovery step 4 only — the documented default, not a contract. The
// port is not the contract; the state file is.
const DefaultPort = 21131

var (
	ErrNoServer   = errors.New("no openjev server reachable")
	ErrNotReady   = errors.New("openjev is running but not ready")
	ErrAPIVersion = errors.New("openjev api version mismatch")
	// ErrTooLong is 422 for input the server will not truncate for us (ADR 0008).
	// Distinct because it is the one error a caller can fix by sending less.
	ErrTooLong = errors.New("input exceeds the server's limits and the server will not truncate it")
)

// Retryable reports whether an error is worth trying again shortly, rather than a
// reason to give up on the backend.
//
// `shutting_down` is the one that is easy to get wrong: it is a 503 from a server
// draining on purpose, not a broken one. Treating it as fatal would turn an ordinary
// restart into "openjev is dead" in the log. Every caller in this plugin degrades to
// pass-through either way, so this exists to keep the REPORTING honest.
func Retryable(err error) bool {
	var ae *APIError
	if errors.As(err, &ae) {
		switch ae.Code {
		case "queue_full", "model_not_ready", "shutting_down", "timeout", "device_error":
			return true
		}
		return false
	}
	return errors.Is(err, ErrNotReady) || errors.Is(err, ErrNoServer)
}

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
	// RetryAfter is the server's own backpressure signal, honoured rather than
	// guessed at. The server runs ONE inference worker (server ADR 0005), so
	// concurrency here is admission control, not parallelism: retrying faster than
	// asked just refills a queue that is already full.
	RetryAfter time.Duration `json:"-"`
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
	// Model is absent until the weights are loaded, which is why it is a pointer:
	// an empty struct would report a server with no model as one running "".
	Model *ModelInfo `json:"model"`
	Phase string     `json:"phase"`
}

// ModelInfo is GET /v1/model, and also rides inside /v1/info once weights are resident.
//
// EntailmentLabel is the load-bearing field. The label set and its ORDER are registry
// config, so "which of these probabilities means entailment" is a question only the
// server can answer. Assuming a name, or an index, yields contradiction probabilities
// dressed as entailment — a confident, exactly-inverted answer that every confidence bar
// in this plugin would wave through.
type ModelInfo struct {
	Model          string   `json:"model"`
	Revision       string   `json:"revision"`
	Device         string   `json:"device"`
	Dtype          string   `json:"dtype"`
	Backend        string   `json:"backend"`
	Context        int      `json:"context"`
	Labels         []string `json:"labels"`
	EntailmentLabel string  `json:"entailment_label"`
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
	entail  string // resolved entailment label, cached for the process lifetime
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

// EntailmentLabel asks the server which score key means entailment.
//
// Cached, because it cannot change under a running server: the label set belongs to the
// loaded model. Resolved from /v1/info's embedded model when the weights are already
// resident, and from /v1/model otherwise — /v1/model is 503 `model_not_ready` until the
// weights load, and that is a real state rather than an error worth failing on.
//
// An empty return is honest and safe: every reader degrades to 0, which refuses an
// answer and discards a classification. The alternative — defaulting to "entailment" —
// is a coin flip on a registry we do not control, and it loses silently.
func (c *Client) EntailmentLabel(ctx context.Context) string {
	if c.entail != "" {
		return c.entail
	}
	if info, err := c.Info(ctx); err == nil && info.Model != nil && info.Model.EntailmentLabel != "" {
		c.entail = info.Model.EntailmentLabel
		return c.entail
	}
	var m ModelInfo
	if err := c.getJSON(ctx, "/v1/model", &m); err == nil {
		c.entail = m.EntailmentLabel
	}
	return c.entail
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

// Fit bounds a string so a request stays inside the server's published limits.
//
// This exists because the server will NOT truncate for us: `truncate: "tail"` is
// refused with 422 (server ADR 0008), on the grounds that character truncation of a
// templated, tokenised pair is a plausible-looking lie. That reasoning is right, and it
// does not go away for us — so Fit is NOT a client-side reimplementation of the thing
// the server declined to do.
//
// The distinction is what the text IS. Fit is only ever applied to material we already
// built as a lossy digest — a pane's recent scrollback, capped and de-noised — where
// "the last N characters" is already the shape of the content and shortening it further
// loses nothing the ranking depended on. It is never applied to a user's question, a
// claim under test, or a caller's option: those go whole or not at all, and an
// over-long one is a 422 the caller must see.
//
// The margin is deliberate. max_field_chars is a CHARACTER bound and the real ceiling is
// tokens, so sitting exactly on it is how a request that measured fine gets refused.
func Fit(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	keep := max - max/8 // ~12% headroom for the template and tokeniser
	if keep <= 0 {
		keep = max
	}
	if len(s) <= keep {
		return s
	}
	return s[len(s)-keep:] // the tail is the recent, diagnostic part
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

// postJSON sends one request, honouring the server's backpressure ONCE.
//
// The server runs a single inference worker behind a bounded queue (server ADR 0005), so
// a 429 means "the queue is full", not "something went wrong". One bounded wait on the
// server's own Retry-After is the honest response: it is the difference between riding
// out a busy moment and giving up on a healthy server. We do not loop — a router that
// retries indefinitely turns admission control into a latency spike, and every caller
// here already degrades to pass-through.
func (c *Client) postJSON(ctx context.Context, path string, body []byte, out any) error {
	err := c.postOnce(ctx, path, body, out)

	var ae *APIError
	if !errors.As(err, &ae) || ae.Code != "queue_full" {
		return err
	}
	wait := ae.RetryAfter
	if wait <= 0 {
		wait = time.Second
	}
	if wait > 5*time.Second {
		wait = 5 * time.Second // the caller has its own budget; never blow it here
	}
	select {
	case <-ctx.Done():
		return err
	case <-time.After(wait):
	}
	return c.postOnce(ctx, path, body, out)
}

func (c *Client) postOnce(ctx context.Context, path string, body []byte, out any) error {
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
			if ra := resp.Header.Get("Retry-After"); ra != "" {
				if secs, perr := strconv.Atoi(ra); perr == nil {
					e.RetryAfter = time.Duration(secs) * time.Second
				}
			}
			if e.Code == "unprocessable" {
				return fmt.Errorf("%w: %s", ErrTooLong, e.Message)
			}
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
	// Consent for a first-ever weight download is CLI-side and requires a TTY
	// (server ADRs 0006/0010). A supervised child is never a TTY, so without this a
	// first run exits 77 with nothing downloaded and no prompt anyone could have
	// answered. Spawning a server IS the decision to let it fetch its weights —
	// making that explicit here is what turns exit 77 from a dead end into a
	// download. An attached server is unaffected: this only ever applies to a child
	// we chose to start.
	cmd.Env = append(os.Environ(), "OPENJEV_ASSUME_YES=1")
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
		// 77 is outside design 02's 0-9 table and means "it wants permission to
		// download" (server ADR 0010). We set OPENJEV_ASSUME_YES above, so seeing
		// it here means the binary is older than that env var — which is a fix a
		// human can act on, not a mystery.
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == ExitConsentRequired {
			return nil, fmt.Errorf("openjev exited 77: it wants consent to download the model, and a supervised child has no TTY to ask. Run `%s model pull --yes` once, or upgrade openjev so it honours OPENJEV_ASSUME_YES", bin)
		}
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

	// EntailmentLabel is which key of Scores means entailment, as the SERVER
	// reported it. It is set by the client from /v1/model, never assumed: the
	// registry chooses both the label names and their order, so neither the string
	// "entailment" nor index 1 is a fact about anything.
	EntailmentLabel string `json:"-"`
}

// Entailment is P(entailment), the calibrated probability every bar in this plugin
// reads as a confidence.
//
// It returns 0 when the label is unknown or absent, and that is deliberate: 0 makes a
// boolean refuse and a classification degrade, whereas any guess here would be an
// inverted answer delivered with full confidence.
func (p Prediction) Entailment() float64 {
	if p.EntailmentLabel == "" {
		return 0
	}
	return p.Scores[p.EntailmentLabel]
}

// Predict runs NLI over premise/hypothesis pairs.
//
// This is how a yes/no question gets a NUMBER instead of a sentence: the task is the
// premise, the claim is the hypothesis, and the entailment probability is the answer
// and its confidence at once. Results come back in request order.
func (c *Client) Predict(ctx context.Context, pairs []Pair) ([]Prediction, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	// truncate is "error", never "tail". The server REFUSES "tail" with 422
	// `unprocessable` (server ADR 0008): core owns tokenisation and exposes no
	// truncating encode, so character-level truncation would be a plausible-looking
	// lie — the template wraps the text, the tokenizer is not a character counter,
	// and getting it slightly wrong yields a confident, wrong, unfalsifiable label.
	//
	// The consequence is ours to carry: the server will not shorten anything, so WE
	// keep requests inside the published limits. See Client.Fit.
	body, _ := json.Marshal(map[string]any{"pairs": pairs, "truncate": "error"})
	var out struct {
		Results []Prediction `json:"results"`
	}
	if err := c.postJSON(ctx, "/v1/predict", body, &out); err != nil {
		return nil, err
	}
	// Stamp the server's own entailment label onto every result, so no caller
	// anywhere has to know — or guess — which key it is.
	label := c.EntailmentLabel(ctx)
	for i := range out.Results {
		out.Results[i].EntailmentLabel = label
	}
	return out.Results, nil
}

// Grade is answer-vs-reference. Used only for an explicitly graded task, never inferred.
type Grade struct {
	Label     string             `json:"label"`
	Scores    map[string]float64 `json:"scores"`
	Pass      bool               `json:"pass"`
	Threshold float64            `json:"threshold"`

	// EntailmentLabel is set by the client from /v1/model, for the same reason as on
	// Prediction. `Pass` is the server's own verdict and is authoritative; this is
	// only for reporting the probability behind it.
	EntailmentLabel string `json:"-"`
}

// Entailment is the probability behind Pass. Zero when the label is unknown.
func (g Grade) Entailment() float64 {
	if g.EntailmentLabel == "" {
		return 0
	}
	return g.Scores[g.EntailmentLabel]
}

func (c *Client) Grade(ctx context.Context, answer, reference string, threshold float64) (Grade, error) {
	var g Grade
	body, _ := json.Marshal(map[string]any{"answer": answer, "reference": reference, "threshold": threshold})
	err := c.postJSON(ctx, "/v1/grade", body, &g)
	g.EntailmentLabel = c.EntailmentLabel(ctx)
	return g, err
}
