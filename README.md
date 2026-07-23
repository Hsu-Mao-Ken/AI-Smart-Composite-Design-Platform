# AI Woven Composite Intelligent Design Platform

## Overview
This repository contains the official implementation of the AI Woven Composite Intelligent Design Platform. The framework seamlessly integrates automated data simulation, a dual-stage surrogate model for macroscopic mechanical property prediction (elasticity and plasticity), and a Proximal Policy Optimization (PPO) algorithm for structural optimization. Furthermore, the platform utilizes Generative AI (Llama-3.1-8B-Instract) to establish an advanced architecture that accepts natural language as the primary input for automated composite material design and evaluation. Unmanned Aerial Vehicles (UAVs) are included as a secondary application-side supplement.

## Repository Structure

### Core Models & Simulation
*   `TexGen_model/`: Scripts for automated mesoscale geometric modeling of woven composites.
*   `Abaqus_model/`: Scripts for automated finite element analysis (FEA) modeling and data generation.
*   `Effective_model/`: The first-stage dual-input Convolutional Neural Network (CNN) for linear elastic property prediction.
*   `Plastic_model/`: The second-stage dual-input CNN for nonlinear plastic response and strain energy density prediction.
*   `PPO_design_model/`: Reinforcement learning framework utilizing PPO for stable and efficient structural generation.

### Secondary Application
*   `UAV_model/` & `UAV_model_load/`: Implementation and loading conditions for the secondary application area focusing on UAV components.

### Platform Application & Weights
*   `LLM_composite_web_PPO_APP.py` & `LLM_composite_web_function_PPO_APP.py`: Main scripts for the web-based Intelligent Design Platform interface.
*   `*.pth`: Pre-trained PyTorch model weights (e.g., `effective_model_pytorch_half.pth`, `plastic_model_pytorch.pth`, `ppo_best_model.pth`).
*   `*.pkl`: Data scalers for normalizing inputs and inverse transforming outputs (e.g., `effective_input_scaler.pkl`).
*   `*.csv`: Material property databases (`fiber_material_property.csv`, `resin_material_property.csv`).

## Hardware & Software Requirements
The computational procedures and model training were conducted using the following setup:
*   **CPU**: Intel Core i7-10700
*   **RAM**: 64 GB
*   **GPU**: NVIDIA GeForce RTX 3090
*   **Framework**: PyTorch (Version 2.9.1+cu130)

Install the required dependencies using:
```bash
pip install -r requirements.txt
