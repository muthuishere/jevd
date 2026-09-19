package cli

import (
	"fmt"
	"os"
	"path/filepath"
)

// pluginRoot is where our files live. Herdr tells a plugin; outside Herdr we fall back
// to the binary's parent so the CLI still works from a checkout.
func pluginRoot() string {
	if r := os.Getenv("HERDR_PLUGIN_ROOT"); r != "" {
		return r
	}
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(filepath.Dir(exe))
}

// skill prints the agent skill, or symlinks it into the skill directories.
//
// SYMLINK, never copy. A copied skill drifts from the binary that implements it, and a
// skill that documents a verb the binary no longer has is worse than no skill at all —
// the agent runs it, it fails, and the agent concludes the whole plugin is broken.
func skill(flags map[string]string) error {
	dir := filepath.Join(pluginRoot(), "skills", "herdr-jev")
	if flags["install"] == "" {
		b, err := os.ReadFile(filepath.Join(dir, "SKILL.md"))
		if err != nil {
			return err
		}
		fmt.Print(string(b))
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	for _, base := range []string{
		filepath.Join(home, ".claude", "skills"),
		filepath.Join(home, ".agents", "skills"),
	} {
		if err := os.MkdirAll(base, 0o755); err != nil {
			return err
		}
		dst := filepath.Join(base, "herdr-jev")
		if err := os.RemoveAll(dst); err != nil {
			return err
		}
		if err := os.Symlink(dir, dst); err != nil {
			return err
		}
		fmt.Printf("linked %s -> %s\n", dst, dir)
	}
	return nil
}
