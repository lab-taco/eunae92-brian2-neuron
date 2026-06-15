# eunae92-brian2-neuron

Personal work repository for Brian2/NEURON simulations and extracellular recording study.

## Goals

1. Study basic computational neuroscience with Neuronal Dynamics.
2. Learn Brian2 for spiking neuron and network simulations.
3. Learn NEURON for biophysical single-cell and compartmental simulations.
4. Summarize extracellular recording principles and practical data analysis methods.

## Environment

Main local environment:

    conda activate brian2-env

Brian2 and NEURON installation were tested in this environment.

## Repository structure

    notebooks/   Jupyter notebooks
    scripts/     Python scripts
    notes/       Reading notes and summaries
    figures/     Saved plots and diagrams
    data/        Local data folder; raw data is ignored by git

## Current notebooks

- notebooks/01_brian2_neuron_test.ipynb: setup check for Brian2 and NEURON
- notebooks/02_brian2_lif_model.ipynb: Brian2 leaky integrate-and-fire model
- notebooks/03_neuron_hh_single_cell.ipynb: NEURON Hodgkin-Huxley single-cell model

## Study roadmap

### Part 1. General computational neuroscience

- Single-neuron dynamics
- Hodgkin-Huxley model
- Integrate-and-fire models
- Synapses and dendrites
- Spike trains and neural coding
- Small recurrent networks

### Part 2. Simulators

- Brian2: units, NeuronGroup, Synapses, StateMonitor, SpikeMonitor
- NEURON: Section, mechanisms, IClamp, Vector recording, hoc/std run
- Compare equivalent models in Brian2 and NEURON

### Part 3. Extracellular recording

- Origin of extracellular potentials
- Spike band vs LFP band
- Filtering and preprocessing
- Spike detection and spike sorting overview
- LFP power spectrum and spectrogram
- Lab-specific data analysis pipeline
