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

func main() { os.Exit(cli.Main(os.Args[1:])) }
