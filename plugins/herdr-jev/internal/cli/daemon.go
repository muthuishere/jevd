package cli

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"syscall"

	"github.com/muthuishere/herdr-jev/internal/config"
)

// daemonCmd fork-execs `serve` and returns immediately.
//
// Herdr's [[startup]] hooks are one-shot and unsupervised: whatever they run is
// expected to return, and a long-lived process started directly from one is a process
// Herdr does not own and will not restart. So the hook runs this, which detaches and
// exits, and the real work happens in `serve`.
func daemonCmd() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(exe, "serve")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	cmd.Stdout, cmd.Stderr = nil, nil
	if err := cmd.Start(); err != nil {
		return err
	}
	_ = cmd.Process.Release()
	return nil
}

// serveCmd is the spawn-mode supervisor, and nothing else.
//
// In `attach` mode it exits 0 immediately: there is no child to own, and a supervisor
// with nothing to supervise is a process that only produces logs. `dispatch` never
// needs this daemon — it is a one-shot that classifies and starts an agent — so the
// daemon's entire job is owning a spawned openjev for as long as Herdr is up.
func serveCmd() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if cfg.OpenJEV.Mode != "spawn" {
		fmt.Println("openjev.mode is \"attach\": nothing to supervise, exiting cleanly")
		return nil
	}

	// The flock is what makes a repeated [[startup]] hook harmless. Herdr runs the
	// hook on every session restore, and without this the third restore would be the
	// third openjev fighting for the same state file.
	lock, err := os.OpenFile(config.PidPath(), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		// Already supervised. Exiting 0 is deliberate: an idempotent start must
		// not look like a failure in Herdr's plugin log.
		fmt.Println("another herdr-jev serve already holds the lock; nothing to do")
		return nil
	}
	_ = lock.Truncate(0)
	fmt.Fprintf(lock, "%d\n", os.Getpid())

	client, err := backend(cfg)
	if err != nil {
		return err
	}
	defer client.Close()
	fmt.Printf("supervising openjev at %s\n", client.URL)

	// Whoever spawned it kills it. Close() sends SIGINT and waits out the grace
	// window; an attached server would never reach here at all.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- client.Wait() }()
	select {
	case <-sig:
		return client.Close()
	case err := <-done:
		return err
	}
}
