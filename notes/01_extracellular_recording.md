# Extracellular recording notes

## Reading list

1. Neuronal Dynamics
2. nrn3241.pdf
3. s41583-026-01042-4.pdf
4. Lab-specific extracellular recording protocol and analysis notes

## Key concepts

### 1. What does an extracellular electrode measure?

- Extracellular voltage
- Transmembrane currents
- Current sources and sinks
- Volume conduction
- Difference between intracellular and extracellular recording

### 2. Signal bands

- Raw broadband signal
- Spike band
- LFP band
- EEG/ECoG/LFP/spikes: spatial and temporal scales

### 3. Basic analysis pipeline

1. Load raw data and metadata
2. Check sampling rate and channel layout
3. Remove bad channels and artifacts
4. Filter into LFP band and spike band
5. Detect spikes or threshold crossings
6. Estimate firing rate
7. Analyze LFP power spectrum and spectrogram
8. Relate signal features to behavior or stimulation

## Questions to answer

- What are the main current sources of extracellular fields?
- Why do synaptic currents often dominate LFP?
- How are spikes and LFP separated analytically?
- What information is lost in extracellular recordings?
- What preprocessing steps does our lab use?
