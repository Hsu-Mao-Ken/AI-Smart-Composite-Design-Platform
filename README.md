# AI Woven Composite Intelligent Design Platform

## Overview
This repository contains the official implementation of the AI Woven Composite Intelligent Design Platform. The framework seamlessly integrates automated data simulation, a dual-stage surrogate model for predicting macroscopic mechanical properties (elasticity and plasticity), and a Proximal Policy Optimization (PPO) algorithm for structural optimization. Furthermore, the platform utilizes a Large Language Model (Llama-3.1-8B-Instruct) to establish an advanced architecture that accepts natural language as the primary input for automated composite material design and evaluation. Unmanned Aerial Vehicles (UAVs) are included as a secondary application-side supplement.

---

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
*   `*.pth`: Pre-trained PyTorch model weights (e.g., `effective_model_pytorch.pth`, `plastic_model_pytorch.pth`, `ppo_best_model.pth`).
*   `*.pkl`: Data scalers for normalizing inputs and inverse transforming outputs.
*   `*.csv`: Material property databases (`fiber_material_property.csv`, `resin_material_property.csv`).

---

##  Hardware & Environment Setup

To ensure full reproducibility, please note that the **Machine Learning / Web Hosting** environment and the **Data Generation (Simulation)** environment operate independently.

**Hardware Baseline used in this study:**
*   **OS**: Windows 11
*   **CPU**: Intel Core i7-10700
*   **RAM**: 64 GB
*   **GPU**: NVIDIA GeForce RTX 3090

### 1. Data Generation Environment (Windows / Abaqus)
The finite element data generation relies on commercial simulation software and is executed within its proprietary, self-contained environment.
*   **Abaqus**: Version 2018 (Utilizes its internal Python 2.7 environment; no external `pip install` is required or supported).
*   **TexGen**: Required for mesoscale geometric modeling.
*   **Note**: While the core FEA logic depends on the internal Abaqus environment, the batch data generation process is orchestrated via standard Python. Specifically, standard Python scripts utilize the `os` module to programmatically issue execution instructions to the Abaqus Command Line, enabling automated, multi-instance simulation runs. 

### 2. Machine Learning & Web Platform Environment (Python / GPU)
This environment is used for training the CNN/PPO models and running the Streamlit web application. We provide a clean, deduplicated `requirements.txt` with fixed versions for visualization tools (`pyvista`, `vtk`, `trame`) and dependencies.

**Step 1: Install PyTorch with explicit CUDA support**
*(Do not rely solely on `requirements.txt` for PyTorch, as CUDA mapping requires specific index URLs based on your local GPU drivers).*
```bash
pip install torch==2.9.1 torchvision==0.24.1 --extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
```
**Step 2: Install the remaining dependencies**
```bash
pip install -r requirements.txt
```
##  Execution & Reproducibility Caveats

While the provided Python scripts in this repository are syntactically valid, **syntax pass does not equate to full end-to-end reproducibility.** Complete execution strictly depends on the configuration of your local host environment. Please ensure the following before execution:

1.  **Hugging Face Access Token (LLM Requirement)**
    *   The `Llama-3.1-8B-Instruct` model requires a verified Hugging Face token. 
    *   **Action Required**: You must generate your own personal access token from Hugging Face and replace the placeholder in `LLM_composite_web_PPO_APP.py` (or set it as an environment variable) before running the LLM module.
2.  **Local File Paths Configuration**
    *   Hardcoded absolute or relative paths in the scripts must be updated to match your local repository directory. 
    *   **Action Required**: Verify and modify the directory paths for `*.pth` (weights), `*.pkl` (scalers), and `*.csv` (databases) within the main scripts based on your local folder structure.
3.  **Commercial Software Dependencies**
    *   Running the automated FEA data generation pipeline (`Abaqus_model/`) requires active Abaqus and TexGen installations/licenses on a Windows machine.

### Starting the Intelligent Design Platform
Once the Hugging Face token, local file paths, and the GPU environment are correctly configured, you can launch the web interface via Streamlit:
```bash
streamlit run LLM_composite_web_PPO_APP.py
```
