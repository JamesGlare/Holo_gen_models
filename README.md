# Familiy of Tensorflow-based models for Conditional Generative Modelling of holograms

The ML models in this repository are used to generative candidates of spatial light modulator (SLM) pixel maps.  
Such pixel maps are sometimes loosely referred to as a type of 'hologram'. The SLM used in this study is of the phase-only variant, which means that it can only affect the phase of incoming wave fronts rather than amplitude and phase.  

The principles underlying phase-only holography, applications and the reasoning behind this approach are described in detail in a forthcoming paper. 

The main models are the cVAE, cGAN and cVAe with foward loss. 
All models are trained on custom datasets of _intensity-hologram_ pairs.

In order to reduce cardinality of the problem, I resorted to training these models on a proxy-mapping rather than the full hologram-to-intensity relation. The details of this proxy mapping and how it relates to the problem of digital SLM-based phase-only holography are described in the paper.
