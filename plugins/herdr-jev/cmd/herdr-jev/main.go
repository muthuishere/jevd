// Command herdr-jev answers a task, routes it, or gets out of the way.
//
// Three stages, in order: a decision-shaped task the local model is sure about is
// ANSWERED on loopback with no agent invoked and nothing leaving the machine; anything
// else has its model tier and reasoning effort chosen by an asymmetric-confidence
// policy and is ROUTED to an agent started with them; and every failure, timeout or
// ambiguity PASSES THROUGH untouched. It never blocks a task.
package main

import (
	"os"

	"github.com/muthuishere/herdr-jev/internal/cli"
)

// Stamped at build time by scripts/build.sh with -ldflags "-X main.version=...".
// A binary that cannot say which build it is cannot be bisected against a bad routing
// decision, and routing decisions are the thing this plugin is judged on.
var (
	version   = "dev"
	commit    = "unknown"
	buildDate = "unknown"
)

func main() {
	cli.Version, cli.Commit, cli.BuildDate = version, commit, buildDate
	os.Exit(cli.Main(os.Args[1:]))
}
