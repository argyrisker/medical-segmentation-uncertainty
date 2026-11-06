# Running the experiments on the HH GPU

One script, `scripts/run_all.sh`, runs the whole plan in order: smoke test,
tuning, final baseline, overfitting study, ablations, uncertainty and the
final test evaluation. You start it once and it keeps going after you log out.

## 1. Log in

Log in the way the "Introduction to HH GPU" lab showed you.

- **SSH (a terminal):** you get a shell prompt on the GPU machine. Continue below.
- **A Jupyter page in the browser:** open *New → Terminal* and type the commands
  there. See the note in step 5 about Jupyter shutting down.

Check that you see a GPU:

```bash
nvidia-smi
```

Note the "CUDA Version" in the top-right corner; step 3 needs it.

## 2. Get the code

```bash
git clone -b claude/busy-goldberg-iefdvb https://github.com/argyrisker/medical-segmentation-uncertainty.git
cd medical-segmentation-uncertainty
```

If the repository is private, GitHub asks for your username and a *personal
access token* (GitHub → Settings → Developer settings → Personal access
tokens), not your password. Alternatively, download the branch as a ZIP from
GitHub on your laptop and upload it.

If you already cloned it earlier, update instead:

```bash
git fetch origin claude/busy-goldberg-iefdvb && git checkout claude/busy-goldberg-iefdvb && git pull
```

## 3. Python environment

Pick the first option that works.

**A. PyTorch is already installed on the machine.** Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints a version and `True`, install only the rest:

```bash
pip install --user monai nibabel scipy scikit-learn pandas matplotlib tqdm pyyaml requests
```

**B. Conda is available** (`conda --version` works):

```bash
bash setup_env.sh gpu
conda activate medseg-uncertainty
```

This installs PyTorch for CUDA 12.8. If `nvidia-smi` showed a lower CUDA
version, use `CUDA_INDEX=cu121 bash setup_env.sh gpu` (or `cu118`) instead.
The script rewrites `requirements.txt` and `environment.yml` at the end; undo
that with `git checkout -- requirements.txt environment.yml`.

**C. Neither: use uv.** Install uv if `uv --version` does not work:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Create the environment, switch it on, and install:

```bash
uv venv ~/medseg
source ~/medseg/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install monai nibabel scipy scikit-learn pandas matplotlib tqdm pyyaml requests
```

After logging in again later, run `source ~/medseg/bin/activate` before any
project command. A job that is already running does not need it.

Whichever you used, this must print `True`:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Start everything

From the repository folder, with the environment active:

```bash
nohup bash scripts/run_all.sh > run_all.log 2>&1 &
```

It downloads the dataset (28 MB) on the first run. If the machine has no
internet access, copy `Task04_Hippocampus.tar` into a `data/` folder in the
repository and extract it there with `tar -xf`.

If the machine uses **Slurm** (the lab used `sbatch`/`squeue`), submit it as a
job instead. Add the partition name your lab used if it needs one:

```bash
sbatch scripts/run_all.sh
```

Options go in front of the command, for example 100 epochs per run and the
second GPU:

```bash
EPOCHS=100 DEVICE=cuda:1 nohup bash scripts/run_all.sh > run_all.log 2>&1 &
```

## 5. Check on it

Which stage it is on:

```bash
grep "########" run_all.log
```

The live output:

```bash
tail -f run_all.log
```

How many training runs have finished (21 in total, including the two smoke-test runs):

```bash
ls outputs/checkpoints/*/summary.json | wc -l
```

**Budget the time after the first real run.** When `v1_legacy` and
`s1a_patch32` have finished, look at how long one took:

```bash
grep elapsed_min outputs/checkpoints/s1a_patch32/summary.json
```

Multiply by 19. If that is more than you have before the slides are due,
stop the job and restart with `EPOCHS=100`. Finished runs are kept.

Stop it:

```bash
pkill -f 'scripts/(run_all|stage|smoke)'; pkill -f src.train
```

On Slurm, use `scancel <job id>` instead (the id is shown by `squeue -u $USER`).

**Jupyter note:** if you started it from a Jupyter terminal, the job may stop
when the Jupyter server shuts down for inactivity. Keep the tab open, or ask
the course assistant whether long jobs should run another way.

**If it crashes or you lose the connection**, start the same command again.
Finished runs are skipped. The error is at the end of `run_all.log` and in
`outputs/logs/<run>.log`.

## 6. Copy the results to your laptop

Pack the tables, metrics and plots (not the model weights):

```bash
tar czf results.tar.gz outputs/tables outputs/metrics reports/figures
```

Then download `results.tar.gz`: right-click → Download in Jupyter, or from
your laptop over SSH:

```bash
scp <user>@<hh-gpu-host>:medical-segmentation-uncertainty/results.tar.gz .
```

What goes on which slide is listed in `plan.md`, section 6.
