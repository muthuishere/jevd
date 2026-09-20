package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
)

// versionCmd prints the build identity. --json because every verb takes --json, and a
// supervisor comparing the running binary against the one on disk should not have to
// parse prose to do it.
func versionCmd(flags map[string]string) error {
	if _, ok := flags["json"]; ok {
		return json.NewEncoder(os.Stdout).Encode(map[string]string{
			"version":   Version,
			"commit":    Commit,
			"buildDate": BuildDate,
			"go":        runtime.Version(),
			"platform":  runtime.GOOS + "/" + runtime.GOARCH,
		})
	}
	fmt.Printf("herdr-jev %s (%s, built %s, %s %s/%s)\n",
		Version, Commit, BuildDate, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	return nil
}
