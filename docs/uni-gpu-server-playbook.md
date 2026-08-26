# Playbook: running a project on the university GPU server via SSH

Hand this file to Claude in a new project and say "read this and connect to
the uni server the same way." It's written as instructions *to Claude*, with
the user's known, reusable facts filled in and per-project blanks marked
`<LIKE THIS>`.

## Known facts about this setup (reusable across projects)

- **Server:** `10.6.32.12` (hostname `aai-lab`), reachable only when the
  user is on the **university VPN** — a bare SSH attempt will time out
  (not "permission denied") if the VPN isn't connected. If you see a
  connection timeout, ask the user to connect the VPN before anything else.
- **Login:** user `sobhan`. Key-based auth is already set up (the user's
  local `~/.ssh/id_ed25519` pubkey is in the server's `authorized_keys`).
  If a fresh machine doesn't have that key registered yet, see "First-time
  key setup" below.
- **Server specs:** RTX 6000 Ada Generation, 49GB VRAM, driver supports up
  to CUDA 13.2, Python 3.12, `git`/`tmux` preinstalled, no job scheduler —
  it's a plain shared SSH box (other lab members may be logged in; check
  `nvidia-smi` for GPU contention before launching a big job).
- **No usage quota** (unlike Colab) — jobs aren't capped or force-disconnected.
  The only constraint is other users' jobs contending for the same GPU.
- The user has **two GitHub identities**: personal (`SobhanSalarian`) and
  `SobhanSalarianCognitivo`. SSH config already has a `github-cognitivo`
  host alias for the second one (see `~/.ssh/config`). If a push to a
  `SobhanSalarianCognitivo` repo fails with "Permission denied" using a
  plain `git@github.com:...` URL, that's this identity conflict — rewrite
  the remote to use `git@github-cognitivo:...` instead.

## Step-by-step: get a project running on the server

1. **Check connectivity first, don't assume.**
   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=8 10.6.32.12 "echo CONNECTED; hostname; nvidia-smi | head -15"
   ```
   - Timeout → VPN isn't connected, ask the user to connect it.
   - `Permission denied (publickey,password)` → key isn't registered on
     this machine yet, see "First-time key setup" below.
   - Success → proceed.

2. **First-time key setup (only if step 1 showed permission denied).**
   Read the local public key and hand the user a one-liner to run in
   *their own* terminal (you can't answer an interactive password prompt):
   ```bash
   cat ~/.ssh/id_ed25519.pub   # or generate one first if it doesn't exist
   ```
   Give the user this to run themselves (PowerShell):
   ```powershell
   Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh sobhan@10.6.32.12 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
   ```
   Then retest step 1.

3. **Get the project's code onto the server.** Prefer a GitHub round-trip
   over scp/rsync — it's the same mechanism the user will use to sync
   results back later, and it matches how they already work.
   - If the local repo has no remote yet, ask which GitHub account it
     should live under, create the repo (user creates it manually on
     github.com if `gh` CLI isn't installed locally — check with
     `gh --version` first), then:
     ```bash
     git remote add origin <git@github.com-or-github-cognitivo:...>
     git push -u origin main
     ```
   - On the server:
     ```bash
     ssh 10.6.32.12 "git clone <repo-url> <project-dir>"
     ```
   - If the repo is **private**, the server needs its own auth (deploy
     key or a token) — ask the user which they'd prefer. If **public**,
     no auth needed for clone.

4. **Environment setup — write a `setup_server.sh`, don't do this by hand.**
   A reusable one-time setup script belongs in the repo (e.g.
   `<project>/setup_server.sh`) so it's reproducible and so a second
   server or a fresh clone doesn't repeat manual debugging. It should:
   - Print `nvidia-smi` so the CUDA version is visible before installing
     anything GPU-specific.
   - Create a venv (`python3 -m venv .venv`), not a global install.
   - Install a **CUDA-matched** PyTorch/framework build — check the
     driver's CUDA version from `nvidia-smi` first, don't assume.
   - Install the project's declared dependencies — **then sanity-check
     for undeclared ones.** Lesson learned: a project's `requirements.txt`
     may be missing packages that "worked" locally only because the local
     Python already had them globally installed (e.g. `scikit-learn` was
     silently required but never listed). Run the actual entry point
     once, end-to-end, before trusting the dependency list.
   - Extract/prepare any bundled data (e.g. a `dataset.zip` shipped
     inside a vendored repo) — don't assume `git clone` alone is enough.
   - End with an explicit sanity check (e.g.
     `python -c "import torch; print(torch.cuda.is_available())"`) and
     don't proceed until that's confirmed `True`.

5. **Long-running jobs go in `tmux`, always.** SSH sessions and their
   child processes die on disconnect; a bare `python train.py` over SSH
   will not survive the user closing their laptop. Launch as:
   ```bash
   ssh 10.6.32.12 "cd <project-dir> && tmux new-session -d -s <session-name> 'bash run_all.sh'"
   ```
   Verify it actually started (not just that the ssh command returned 0):
   ```bash
   sleep 15 && ssh 10.6.32.12 "tmux list-sessions"
   ```
   A tmux session with no output/immediately gone means the command
   inside it crashed instantly — check the log file, don't assume success.

6. **Monitor via log file + `Monitor` tool, not by repeatedly SSHing in
   yourself.** Have the run script write structured, greppable progress
   lines (e.g. `[dataset] fold N done in Xs (metric=Y)`) to a log file on
   the server. Then watch it remotely with something like:
   ```bash
   ssh 10.6.32.12 "tail -f -n +1 ~/<project-dir>/results/run.log" | grep -E --line-buffered "<stage-transition-pattern>|Traceback|Error|complete"
   ```
   Use the `Monitor` tool (`persistent: true` for a job that runs for
   hours) so you get pushed notifications instead of polling. Keep the
   grep filter to *transitions and errors*, not every line — a
   fold/epoch/step-level filter is fine for a periodic (e.g. every 30 min)
   status snapshot into a local log file, but don't fire a notification
   per step for a multi-hour job.

7. **If something's running slower than expected, verify before
   theorizing.** Don't just assume a known bug reappeared or a new one
   exists — check directly:
   ```bash
   ssh 10.6.32.12 "grep -n '<the exact fixed line>' <path/to/patched/file>"   # confirm a known fix is actually present
   ssh 10.6.32.12 "nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv && top -bn1 | head -12"
   ```
   GPU% + per-core CPU% together usually tell you whether you're
   GPU-bound, CPU-bound (e.g. a single-threaded Python preprocessing
   loop), or actually stalled.

8. **Retrieving results.** Prefer committing results back through git
   from the server (matches how the project already tracks results) over
   manual `scp`:
   ```bash
   ssh 10.6.32.12 "cd <project-dir> && git add <results-path> && git commit -m '...' && git push"
   ```
   then `git pull` locally. Only fall back to `scp -r` for files that
   shouldn't go into git (large raw logs, checkpoints, etc.).

## Things that went wrong last time (so you don't repeat them)

- Assumed a `git clone` of a vendored upstream repo would include its
  data — it didn't; the data was zipped inside the repo and needed an
  explicit `unzip`.
- Assumed the official code's `requirements.txt` was complete — it was
  missing `scikit-learn`, invisible locally because it happened to
  already be installed globally.
- Wrote local setup/run scripts and pushed the *project* to GitHub before
  committing the *scripts themselves* — the server's first `git pull`
  didn't have them yet. Commit and push tooling changes together with
  (or before) telling the server to pull.
- Two training modes (GIN vs. GraphSAGE) writing to the same output
  filename would have silently overwritten each other's results — added
  an explicit `--tag`/suffix argument once this was noticed. Worth
  checking for this kind of collision whenever a script is reused with
  different flags across the same dataset/input.
