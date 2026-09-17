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

### Quiet Training (25W Power Limit)
```bash
python control.py --full --gpu --gpu-power 25
```

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
python control.py --full --gpu --pool 300 --selected 100 --gpu-power 25
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
--gpu-power W          GPU power limit in Watts (5-50W, no limit if not set)
--gpu-memory-fraction F GPU memory fraction (default: 0.4)
```

---

### `train_agents.py` - Individual Agent Training

#### Train Specific Agents
```bash
# Train agents 0-5
python train_agents.py train 0 5

# Train with custom generations and timesteps
python train_agents.py train 0 10 --generations 50 --time-steps 500

# Train with quiet GPU
python train_agents.py train 0 5 --gpu --gpu-power 20
```

#### Build Quality Pool
```bash
# Build pool of 100 agents (resampling those below min-fitness)
python train_agents.py build-pool 100

# With custom parameters
python train_agents.py build-pool 100 --min-fitness 1.0 --max-attempts 8 --gpu-power 25
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
--gpu                  Use GPU
--cpu                  Force CPU
--gpu-power W          GPU power limit in Watts
--gpu-memory-fraction F GPU memory fraction
```

---

## GPU Power Management

### Why Use `--gpu-power`?
- **Reduce fan noise** during training
- **Lower heat** output
- **Save battery** on laptops
- **Trade-off**: Lower power = slower training

### Recommended Settings

| Power Limit | Use Case | Behavior |
|-------------|----------|----------|
| 5W | Ultra silent | Very slow, minimal heat |
| 15W | Very quiet | Silent fans, acceptable speed |
| **25W** | **Quiet + Fast (Recommended)** | **Balanced, minimal fan noise** |
| 35W | Normal | Default, standard speed/heat |
| 45W+ | Performance | Full speed, may trigger fans |

### Examples
```bash
# Silent training
python control.py --full --gpu --gpu-power 15

# Balanced (recommended for most users)
python control.py --full --gpu --gpu-power 25

# Full performance
python control.py --full --gpu --gpu-power 35
```

### Manual GPU Power Control
If `--gpu-power` isn't available or you need manual control:

```bash
# Check current power
nvidia-smi -i 0 -q -d POWER | grep "Current Power Limit"

# Set power limit (requires nvidia-smi)
nvidia-smi -i 0 -pl 25

# Monitor during training
watch -n 1 'nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv,noheader'
```

---

## Hardware Specifications

### GPU: NVIDIA RTX 2000 Ada Laptop GPU
- Max Power: 50W
- Default Power: 35W
- Min Power: 5W
- Idle Power: ~8-9W
- Max Clocks: 3.80 GHz GPU, 8001 MHz Memory
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
python control.py --full --gpu --gpu-power 25 --pool 50 --selected 30 --generations 30
```

### Full Paper-Scale Experiment
```bash
python control.py --full --gpu --gpu-power 35
```

### Silent Full Experiment
```bash
python control.py --full --gpu --gpu-power 20 --pool 300 --selected 100
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

### GPU Power Limit Fails
```bash
# nvidia-smi requires GPU to be initialized first
# Run control.py without --gpu-power first to initialize
# Then set power manually:
nvidia-smi -i 0 -pl 25
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

# Increase power limit
python control.py --full --gpu --gpu-power 35

# Or reduce simulation timesteps
python control.py --full --gpu --time-steps 500
```

---

## Performance Tips

1. **Use `--gpu-power 25`** for best balance of speed and quiet operation
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
python control.py --full --gpu --gpu-power 25
```

---

## See Also
- `COMMANDS.md` in `~/Projects/Dotfiles/UI/` - Low-level power management commands
- `../CLAUDE.md` - Project architecture and terminology
- `simulation.py` - Core simulation parameters
- `generate.py` - Agent/connector generation logic
