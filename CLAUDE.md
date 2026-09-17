# Evolutionary Robotics Project

## Project Overview
This project contains code for Evolutionary Robotics experiments, focusing on evolving robot morphologies and controllers through simulation and optimization.

## Core Architecture
- **Simulation Engine:** PyBullet and Taichi for physics simulation
- **Key Directories:**
  - `src/classes/`: Core object definitions (Robot, HillClimber, etc.)
  - `src/pybullet/`: Physics simulation wrappers
  - `src/taichi/`: High-performance computing components
- **Optimization:** Hill climbing and genetic algorithms for evolving robot parameters

## Key Terminology
- **Fitness:** The distance a robot travels in the simulation
- **Genotype:** The underlying parameters defining the robot (weights, lengths)
- **Phenotype:** The physical manifestation of the robot in the simulator

## General Standards
- Prefer minimal, focused code
- Avoid over-engineering solutions
- Keep implementations simple and direct

## Taichi Module (`src/taichi/`)
This section models robots made of springs using Taichi's differential physics simulator.

### Goals
- Train 2D spring-based robots to move left
- Leverage Taichi's differential simulation capabilities for gradient-based optimization

### Standards for Taichi Code
- Reference the [Taichi documentation](https://docs.taichi-lang.org/) when needed
- Focus on performance through Taichi's GPU-accelerated kernels
- Use differentiable programming for optimization
