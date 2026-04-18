# 📊 PSA-PSO Data Repository

This repository contains the **datasets, benchmark definitions, implementation files, and experimental results** for the research paper:

> **Deep Learning Surrogate-Assisted Particle Swarm Optimization in Partitioned Search Spaces with Adaptive Neighborhood Topology for Continuous Optimization (PSA-PSO)**

---

## 📌 Overview

This repository is provided to ensure **reproducibility, transparency, and validation** of the proposed **PSA-PSO algorithm**.

It includes:
- Benchmark datasets (shifted functions)
- Full experimental results
- Core PSA-PSO implementation
- Surrogate neural network model
- Supporting benchmark utilities

---

## 📁 Repository Structure

| File | Description |
|------|------------|
| `Results.xlsx` | Contains all experimental results reported in the paper, including performance metrics such as mean, standard deviation, and success rate across benchmark functions. |
| `ai_pso_2s.py` | Main implementation of the PSA-PSO algorithm for standard benchmark functions. |
| `ai_pso_2s_l.py` | PSA-PSO implementation adapted for the Lennard-Jones optimization problem. |
| `benchmark.py` | Contains definitions of all benchmark functions used in the experiments (Sphere, Rosenbrock, Ackley, Rastrigin, Griewank). |
| `surrogate.py` | Implementation of the deep learning surrogate model (MLP) used to approximate objective functions within partitioned regions. |
| `sphere_func_data.txt` | Shifted Sphere function dataset used for evaluation. |
| `rosenbrock_func_data.txt` | Shifted Rosenbrock function dataset used for evaluation. |
| `ackley_func_data.txt` | Shifted Ackley function dataset used for evaluation. |
| `rastrigin_func_data.txt` | Shifted Rastrigin function dataset used for evaluation. |
| `griewank_func_data.txt` | Shifted Griewank function dataset used for evaluation. |

---

## 🧠 Method Summary

The **PSA-PSO algorithm** integrates:

- **Search Space Partitioning**  
  The global search domain is divided into structured regions to enhance exploration.

- **Deep Learning Surrogate Model**  
  A neural network (MLP) approximates the objective function locally to guide optimization.

- **Adaptive Neighborhood Topology**  
  A dynamic ring-based topology that progressively increases connectivity between particles.

- **Hybrid Optimization Strategy**  
  Combines surrogate-guided local search with global PSO exploration.

---

## 🧪 Benchmarks

Experiments were conducted on standard continuous optimization functions:

- Sphere  
- Rosenbrock  
- Ackley  
- Rastrigin  
- Griewank  

All benchmark functions include **shift transformations** to increase problem difficulty and avoid bias toward the origin.

---

## ▶️ Usage

### Run PSA-PSO on Benchmark Functions
```bash
python ai_pso_2s.py
