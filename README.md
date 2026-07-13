# Cloud-Native Physics-Informed Hybrid Quantum Predictive Maintenance System

A hybrid quantum-classical machine learning system designed to predict imminent mechanical failures using the NASA CMAPSS turbofan dataset. The pipeline introduces physics-informed numerical differentiation into classical baselines and prepares the architecture for remote cloud execution via Amazon Braket Hybrid Jobs.

## 🛠️ System Architecture & Workflow
1. **Physics-Informed Feature Engineering**: Sensor streams are smoothed using rolling metrics, followed by finite-difference differential calculus to extract physical dynamics: $\frac{dT}{dt} \approx \frac{T_t - T_{t-1}}{\Delta t}$ and $\frac{dv}{dt} \approx \frac{v_t - v_{t-1}}{\Delta t}$
2. **Classical Baselines**: Engineered features are evaluated against Random Forest and XGBoost classifiers to establish high-performance boundaries.
3. **Quantum Variational Layer**: Normalised data is amplitude-mapped onto an optimized 4-qubit parameterized variational quantum circuit (VQC) designed using PennyLane.
4. **Cloud Scalability**: Containerized logic prepared for AWS execution using an `AwsQuantumJob` harness targeting the SV1 Amazon Braket Simulator.

## 📊 Performance & Benchmarks
Our feature-engineering framework achieved strong classical baselines in under a second:
* **XGBoost F1-Score**: 80.10%
* **XGBoost Accuracy**: 94.40%
* **Inference Speed**: 0.5619 seconds

## 🚀 Deployment Structure
The cloud pipeline is fully containerized. Running `aws/submit_braket_job.py` configures a global AWS authentication layer and leverages the Amazon Braket SDK to instantiate an autonomous hybrid training task offloading to the remote SV1 state-vector simulator.
