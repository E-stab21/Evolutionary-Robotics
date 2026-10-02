# Replication Project Usage Guide

## Overview
This project runs evolutionary experiments to train and select spring-based robot morphologies. It includes agent generation, training, selection, connector generation, and collective deployment.

---

## Quick Start

### Default Run (Light/CPU)
```bash
python control.py
```

### Full Experiment with GPU
```bash
python control.py --full --gpu
```

### Quiet Training (Recommended: `--pause`)
```bash
python control.py --full --gpu --pause 0.75
```
No sudo needed, and this is the approach that's actually worked well in practice — see [Fan Noise / Cooling](#fan-noise--cooling) below.

---

## Main Commands

### `control.py` - Experiment Pipeline
Runs the full experiment: generate agents → train → select → generate connectors → train connectors → deploy

```bash
# Light run (default: 12 agents, 8 selected, CPU)
python control.py

# Full run with GPU
python control.py --full --gpu

# Full run with custom settings
python control.py --full --gpu --pool 300 --selected 100
```

**Options:**
```
--full                 Run full scale (300 agents, 100 selected, 40 connectors)
--gpu                  Use NVIDIA GPU (default: CPU)
--cpu                  Force CPU mode
--pool N               Number of agents to generate (default: 12 light / 300 full)
--selected N           Number to select (default: 8 light / 100 full)
--connectors-per-set N Connectors per set (default: 1 light / 10 full)
--samples N            Deployment samples per agent (default: 1 light / 20 full)
--generations N        Training generations (default: 2 light / 100 full)
--time-steps N         Simulation steps (default: 80 light / 1000 full)
--min-fitness F        Reject morphologies below this (default: 0.2 light / 1.0 full)
--max-attempts N       Tries per agent (default: 3 light / 8 full)
--pause S              Seconds to sleep after each generation (default: 0, no pause)
--gpu-memory-fraction F GPU memory fraction (default: 0.4)
```

See [Fan Noise / Cooling](#fan-noise--cooling) below for `--pause` and other options to keep things quiet.

---

### `train_agents.py` - Individual Agent Training

#### Train Specific Agents
```bash
# Train agents 0-5
python train_agents.py train 0 5

# Train with custom generations and timesteps
python train_agents.py train 0 10 --generations 50 --time-steps 500

# Train with quiet GPU (sleep 0.5s after each generation to let it cool)
python train_agents.py train 0 5 --gpu --pause 0.5
```

#### Build Quality Pool
```bash
# Build pool of 100 agents (resampling those below min-fitness)
python train_agents.py build-pool 100

# With custom parameters
python train_agents.py build-pool 100 --min-fitness 1.0 --max-attempts 8
```

#### Watch Agent Playback
```bash
# Visualize trained agent
python train_agents.py watch 0

# Watch agent 5 at 30 FPS
python train_agents.py watch 5
```

**Options:**
```
--directory DIR        Agent storage directory (default: agents)
--generations N        Training generations
--time-steps N         Simulation timesteps per generation
--min-fitness F        Reject morphologies below this fitness
--max-attempts N       Max resampling attempts
--pause S              Seconds to sleep after each generation (default: 0, no pause)
--gpu                  Use GPU
--cpu                  Force CPU
--gpu-memory-fraction F GPU memory fraction
```

---

## Fan Noise / Cooling

### `--pause` (Recommended)
Built into `control.py` and `train_agents.py` — no sudo, no hardware fighting. After each training generation, it sleeps for at least `--pause` seconds (or half that generation's compute time, whichever is longer). This gives the GPU real idle time between bursts instead of sustained load, so it never ramps hard enough to spin the fans up in the first place. This is the option that's actually worked well here in practice.

```bash
python control.py --full --gpu --pause 0.75
python train_agents.py build-pool 300 --gpu --pause 0.5
```

| `--pause` | Use Case | Behavior |
|-----------|----------|----------|
| 0 (default) | Normal | No throttling, fastest, loudest |
| 0.25-0.5 | Quiet | Noticeably less fan activity |
| **0.75-1.0** | **Very quiet (recommended)** | **Fans mostly stay down, meaningfully slower** |

### Why not GPU power/clock limits?
This GPU is an RTX 2000 Ada *Laptop* GPU (Dell Precision 5490), and on this hardware `nvidia-smi`'s direct power/thermal controls are locked out at the firmware level — confirmed for all of:
- `-pl` (power limit) — "not supported in current scope"
- `-gtt` (target temperature) — "not supported"
- `-ac` (application clocks) — "deprecated"

None of these work even with `sudo` — it's not a permissions issue, the vBIOS just doesn't expose them on this part.

The one exception is `-lgc` (lock GPU clocks), which is a genuine software-level, root-gated control that actually works here. It caps the graphics clock ceiling, which indirectly reduces power draw:

```bash
sudo nvidia-smi -i 0 -lgc 210,1200   # cap graphics clock to 1200 MHz
python control.py --full --gpu
sudo nvidia-smi -i 0 -rgc            # reset when done
```

It's a secondary option — `--pause` doesn't need root and has been the more effective lever in practice, since a locked-but-sustained clock still keeps the GPU busy (and can still trigger fans), whereas `--pause` gives it real recovery time.

---

## Hardware Specifications

### GPU: NVIDIA RTX 2000 Ada Laptop GPU
- Max Power: 50W (fixed — `nvidia-smi -pl`/`-gtt`/`-ac` are all firmware-locked on this laptop GPU, see [Fan Noise / Cooling](#fan-noise--cooling))
- Default Power: 35W
- Min Power: 5W
- Idle Power: ~8-9W
- Max Graphics Clock: 3105 MHz, Memory 8001 MHz
- Memory: 8 GB GDDR6

### CPU: Intel Core (24 cores)
- Max Frequency: 3.80 GHz
- Min Frequency: 400 MHz
- Governor: powersave (default, optimal for quiet operation)

---

## Configuration Examples

### Light Development (Fast Iteration)
```bash
python control.py --pool 5 --selected 3 --samples 1 --generations 5 --time-steps 100
```

### Medium Training (Balanced)
```bash
python control.py --full --gpu --pool 50 --selected 30 --generations 30 --pause 0.5
```

### Full Paper-Scale Experiment
```bash
python control.py --full --gpu
```

### Silent Full Experiment
```bash
python control.py --full --gpu --pool 300 --selected 100 --pause 1.0
```

---

## Output and Results

### Generated Files
- `agents_pool/`: Quality-filtered morphologies
- `agents/`: Selected morphologies for experiments
- `connectors/`: Connector morphologies (weak/strong, uniform/diverse)
- `results.csv`: Deployment results (fitness across environments)

### Example Results Structure
```
agents_pool/agent0.npz      # Generated morphology
agents_pool/agent0.best.npz # Best training weights
agents/agent5.npz           # Selected morphology
connectors/connector0.npz   # Connector morphology
results.csv                 # Final deployment results
```

---

## Troubleshooting

### GPU Not Being Used
```bash
# Verify GPU is available
nvidia-smi

# Force GPU usage
python control.py --full --gpu

# Check if CUDA is working
python -c "import taichi as ti; ti.init(arch=ti.cuda)"
```

### GPU Power/Thermal Limits (`-pl`, `-gtt`, `-ac`) Fail
```
Changing power management limit is not supported in current scope for GPU: ...
```
Expected on this laptop GPU — these are locked out in firmware regardless of privilege. Use `--pause` instead (see [Fan Noise / Cooling](#fan-noise--cooling)), or `-lgc`/`-rgc` if you specifically want a clock cap:
```bash
sudo nvidia-smi -i 0 -lgc 210,1200

# Confirm it took effect
nvidia-smi -i 0 --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv,noheader
```

### Out of Memory
```bash
# Reduce GPU memory fraction
python control.py --full --gpu --gpu-memory-fraction 0.3

# Or reduce problem size
python control.py --pool 100 --selected 50
```

### Slow Training
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Lower (or remove) --pause
python control.py --full --gpu --pause 0

# Or reduce simulation timesteps
python control.py --full --gpu --time-steps 500
```

---

## Performance Tips

1. **Use `--pause 0.75`** for best balance of speed and quiet operation (no sudo needed)
2. **Use `--full`** flag for production runs (more agents = better statistics)
3. **Monitor with `nvidia-smi -l 1`** during first run to check temperatures
4. **Keep GPU under 70°C** for optimal performance and longevity
5. **Run overnight** for full experiments (can take several hours)

---

## Environment Setup

```bash
# Activate virtual environment
source ~/.venv/bin/activate

# Navigate to replication directory
cd ~/Projects/Evolutionary-Robotics/src/taichi/replication

# Run experiment
python control.py --full --gpu
```

---

## See Also
- `COMMANDS.md` in `~/Projects/Dotfiles/UI/` - Low-level power management commands
- `../CLAUDE.md` - Project architecture and terminology
- `simulation.py` - Core simulation parameters
- `generate.py` - Agent/connector generation logic
