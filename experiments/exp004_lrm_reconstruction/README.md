# Experiment 004 - Zero-shot LRM reconstruction

Inputs are deterministic 512x512 RGBA renders from four azimuths per
Smithsonian reference asset. The first baseline is the official MIT-licensed
TripoSR implementation and checkpoint. Each view is reconstructed independently
to expose view-dependent geometry and provide a direct uncertainty signal.

The vendored TripoSR source is retained under its original license. Its
`torchmcubes` import has a documented scikit-image CPU marching-cubes fallback
because no compatible Windows/Python 3.12 binary was available in this
environment; neural inference remains on CUDA.

