package jev

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// These tests run against a fake that mimics the REAL server's shapes, taken from
// crates/openjev-cli/src/api.rs — not against a convenient one.
//
// That distinction is the point. A fake that is more permissive than the server is
// exactly how a wrong-answer bug survives a green suite: it answers what we hoped for
// rather than what the server sends, and the first real request is the first test.
// So: scores keyed by the REGISTRY's label names (not "entailment"), truncate "tail"
// refused with 422, and the documented error envelope on every failure.

// registryLabels are deliberately NOT the obvious names. The label set and its order are
// registry config, so any code that passes with these and fails with ("contradiction",
// "entailment", "neutral") was reading a guess.
var registryLabels = []string{"contra", "entails", "neither"}

const entailLabel = "entails"

type fakeServer struct {
	entailmentLabel string
	modelLoaded     bool
	predict         map[string]float64 // hypothesis -> P(entailment)
	rerank          []float64
	failNext        *APIError
	seenTruncate    string
	calls           int
}

func (f *fakeServer) start(t *testing.T) *Client {
	t.Helper()
	mux := http.NewServeMux()

	fail := func(w http.ResponseWriter, e *APIError) {
		if e.RetryAfter > 0 {
			w.Header().Set("Retry-After", "1")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(e.Status)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"code": e.Code, "message": e.Message, "request_id": "01TEST"}})
	}

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "phase": "ready"})
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"ready": true, "phase": "ready"})
	})
	mux.HandleFunc("/v1/info", func(w http.ResponseWriter, _ *http.Request) {
		body := map[string]any{
			"object": "info", "api_version": 1, "server_version": "0.1.0", "phase": "ready",
			"capabilities": []string{"predict", "rerank", "grade", "events"},
			"limits": map[string]any{
				"max_options": 4, "max_field_chars": 32768, "max_pairs": 256, "max_queue": 64},
		}
		// /v1/info omits `model` entirely until weights are resident.
		if f.modelLoaded {
			body["model"] = f.modelInfo()
		}
		_ = json.NewEncoder(w).Encode(body)
	})
	mux.HandleFunc("/v1/model", func(w http.ResponseWriter, _ *http.Request) {
		if !f.modelLoaded {
			fail(w, &APIError{Code: "model_not_ready", Status: 503, Message: "weights not resident"})
			return
		}
		_ = json.NewEncoder(w).Encode(f.modelInfo())
	})
	mux.HandleFunc("/v1/predict", func(w http.ResponseWriter, r *http.Request) {
		f.calls++
		if f.failNext != nil {
			e := f.failNext
			f.failNext = nil
			fail(w, e)
			return
		}
		var req struct {
			Pairs []struct {
				Premise    string `json:"premise"`
				Hypothesis string `json:"hypothesis"`
				ID         string `json:"id"`
			} `json:"pairs"`
			Truncate string `json:"truncate"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		f.seenTruncate = req.Truncate

		// The server REFUSES "tail" with 422 (server ADR 0008). A fake that
		// accepted it would hide the one bug this test exists to catch.
		if req.Truncate == "tail" {
			fail(w, &APIError{Code: "unprocessable", Status: 422,
				Message: `truncate: "tail" is not supported; core exposes no truncating encode`})
			return
		}
		results := []map[string]any{}
		for i, p := range req.Pairs {
			ent := f.predict[p.Hypothesis]
			results = append(results, map[string]any{
				"id": p.ID, "index": i, "label": entailLabel,
				"scores": map[string]float64{
					registryLabels[0]: (1 - ent) / 2,
					entailLabel:       ent,
					registryLabels[2]: (1 - ent) / 2,
				}})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "predict", "results": results})
	})
	mux.HandleFunc("/v1/rerank", func(w http.ResponseWriter, r *http.Request) {
		f.calls++
		var req struct {
			Question string   `json:"question"`
			Options  []string `json:"options"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Options) > 4 {
			fail(w, &APIError{Code: "payload_too_large", Status: 413, Message: "too many options"})
			return
		}
		results := []map[string]any{}
		for i := range req.Options {
			s := 0.0
			if i < len(f.rerank) {
				s = f.rerank[i]
			}
			results = append(results, map[string]any{"rank": i, "index": i, "score": s})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "rerank", "results": results})
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return &Client{URL: srv.URL, HTTP: &http.Client{Timeout: 5 * time.Second}}
}

func (f *fakeServer) modelInfo() map[string]any {
	label := f.entailmentLabel
	if label == "" {
		label = entailLabel
	}
	return map[string]any{
		"model": "openjev/test", "revision": "v1", "device": "cpu", "dtype": "f16",
		"backend": "llama.cpp", "context": 4096,
		"labels":  registryLabels, "entailment_label": label,
	}
}

func TestPredictNeverAsksForTailTruncation(t *testing.T) {
	// The server refuses "tail" with 422 because character truncation of a templated,
	// tokenised pair would be a plausible-looking lie. Asking for it turns every
	// long-input request into an error.
	f := &fakeServer{modelLoaded: true, predict: map[string]float64{"x": 0.9}}
	c := f.start(t)
	if _, err := c.Predict(context.Background(), []Pair{{Premise: "a", Hypothesis: "x"}}); err != nil {
		t.Fatalf("predict failed: %v", err)
	}
	if f.seenTruncate != "error" {
		t.Errorf("truncate = %q, want %q — \"tail\" is refused by the real server", f.seenTruncate, "error")
	}
}

func TestEntailmentIsReadByTheServersOwnLabelNotAGuess(t *testing.T) {
	// The whole point: this registry calls it "entails". Any code reading "entailment"
	// or index 1 gets zero here — or, on a differently ordered registry, gets the
	// CONTRADICTION probability dressed as entailment.
	f := &fakeServer{modelLoaded: true, predict: map[string]float64{"claim": 0.87}}
	c := f.start(t)

	if got := c.EntailmentLabel(context.Background()); got != entailLabel {
		t.Fatalf("EntailmentLabel = %q, want %q", got, entailLabel)
	}
	preds, err := c.Predict(context.Background(), []Pair{{Premise: "p", Hypothesis: "claim"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := preds[0].Entailment(); got < 0.86 || got > 0.88 {
		t.Errorf("Entailment() = %v, want ~0.87 read via the %q key", got, entailLabel)
	}
	if _, ok := preds[0].Scores["entailment"]; ok {
		t.Fatal("this fake must NOT contain a literal \"entailment\" key, or it cannot catch the bug")
	}
}

func TestAnUnresolvedLabelReadsAsZeroRatherThanAGuess(t *testing.T) {
	// /v1/model is 503 until weights are resident. Zero makes a boolean refuse and a
	// classification degrade; any default would be an inverted answer at full
	// confidence.
	f := &fakeServer{modelLoaded: false, predict: map[string]float64{"claim": 0.87}}
	c := f.start(t)
	preds, err := c.Predict(context.Background(), []Pair{{Premise: "p", Hypothesis: "claim"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := preds[0].Entailment(); got != 0 {
		t.Errorf("Entailment() = %v, want 0 when the label is unknown", got)
	}
}

func TestErrorCodesAreClassifiedTheWayTheServerMeansThem(t *testing.T) {
	cases := []struct {
		name      string
		err       *APIError
		retryable bool
		tooLong   bool
	}{
		// Beyond design 02's table, and a 503 from a server draining ON PURPOSE.
		// Treating it as fatal turns an ordinary restart into "openjev is dead".
		{"shutting_down is retryable", &APIError{Code: "shutting_down", Status: 503}, true, false},
		{"model_not_ready is retryable", &APIError{Code: "model_not_ready", Status: 503}, true, false},
		{"unprocessable is the caller's to fix", &APIError{Code: "unprocessable", Status: 422}, false, true},
		{"invalid_request is not retryable", &APIError{Code: "invalid_request", Status: 400}, false, false},
		{"internal is not retryable", &APIError{Code: "internal", Status: 500}, false, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := &fakeServer{modelLoaded: true, failNext: c.err}
			cl := f.start(t)
			_, err := cl.Predict(context.Background(), []Pair{{Premise: "a", Hypothesis: "b"}})
			if err == nil {
				t.Fatal("expected an error")
			}
			if c.tooLong {
				if !errors.Is(err, ErrTooLong) {
					t.Errorf("422 should surface as ErrTooLong, got %v", err)
				}
				return
			}
			if Retryable(err) != c.retryable {
				t.Errorf("Retryable(%v) = %v, want %v", err, !c.retryable, c.retryable)
			}
		})
	}
}

func TestRetryableClassifiesTheCodesDirectly(t *testing.T) {
	// queue_full is classified here rather than through a round trip, because a round
	// trip now SUCCEEDS on the retry — which is the behaviour, not a gap.
	retryable := []string{"queue_full", "model_not_ready", "shutting_down", "timeout", "device_error"}
	for _, code := range retryable {
		if !Retryable(&APIError{Code: code}) {
			t.Errorf("%s should be retryable", code)
		}
	}
	for _, code := range []string{"invalid_request", "unauthorized", "not_found", "payload_too_large", "internal"} {
		if Retryable(&APIError{Code: code}) {
			t.Errorf("%s should not be retryable", code)
		}
	}
}

func TestQueueFullIsRetriedExactlyOnce(t *testing.T) {
	// The server runs ONE inference worker behind a bounded queue, so 429 means "busy",
	// not "broken". One bounded wait rides it out; looping would turn admission control
	// into a latency spike.
	f := &fakeServer{modelLoaded: true, predict: map[string]float64{"b": 0.9},
		failNext: &APIError{Code: "queue_full", Status: 429, RetryAfter: time.Second}}
	c := f.start(t)
	preds, err := c.Predict(context.Background(), []Pair{{Premise: "a", Hypothesis: "b"}})
	if err != nil {
		t.Fatalf("a single queue_full should have been ridden out, got %v", err)
	}
	if len(preds) != 1 {
		t.Fatalf("got %d predictions", len(preds))
	}
	if f.calls != 2 {
		t.Errorf("server saw %d calls, want exactly 2 (one rejected, one retry)", f.calls)
	}
}

func TestRerankChunksToTheServersPublishedMaxOptions(t *testing.T) {
	// max_options is 4 on this fake, and exceeding it is a 413. Chunking to the
	// PUBLISHED limit rather than a hardcoded constant is what keeps that from being
	// discovered in production.
	f := &fakeServer{modelLoaded: true, rerank: []float64{0.1, 0.2, 0.3, 0.4}}
	c := f.start(t)
	opts := make([]string, 10)
	for i := range opts {
		opts[i] = "option"
	}
	scored, err := c.Rerank(context.Background(), "q", opts)
	if err != nil {
		t.Fatalf("rerank should have chunked, got %v", err)
	}
	if len(scored) != 10 {
		t.Fatalf("got %d scores for 10 options", len(scored))
	}
	// Indices must be translated back to the CALLER's positions, or a second chunk
	// silently overwrites the first's answers.
	seen := map[int]bool{}
	for _, s := range scored {
		if s.Index < 0 || s.Index > 9 {
			t.Fatalf("index %d is outside the caller's slice", s.Index)
		}
		if seen[s.Index] {
			t.Fatalf("index %d appeared twice: chunk offsets are not being applied", s.Index)
		}
		seen[s.Index] = true
	}
}

func TestFitLeavesShortTextAloneAndKeepsTheTail(t *testing.T) {
	if got := Fit("short", 100); got != "short" {
		t.Errorf("Fit should not touch text inside the limit, got %q", got)
	}
	long := strings.Repeat("a", 100) + "TAIL"
	got := Fit(long, 50)
	if len(got) >= 50 {
		t.Errorf("Fit left %d chars against a 50 limit; the margin covers the template", len(got))
	}
	if !strings.HasSuffix(got, "TAIL") {
		t.Error("Fit must keep the tail: it is the recent, diagnostic part")
	}
}
