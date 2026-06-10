# LeS1mple: A Hierarchical JEPA World Model for Long-Horizon Multi-Agent Dynamics in Counter-Strike

LeS1mple is world model project for Counter-Strike. It uses the [OpenCS2 Dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset), which provides thousands of hours of tick-aligned video, audio, input actions, and world state. The goal is to learn a predictive, latent state-action representation of gameplay that can be used for planning. 

The name is a portmanteau of S1mple, arguably the greatest CS player ever, and [LeWorldModel](https://arxiv.org/abs/2603.19312), a Joint Embedding Predictive Archicetcture (JEPA) that leverages a stable, effective, and very simple training objective.  

I chose CS because it is challenging partially observable multi-agent environment that requires complex strategy, quick reaction times, and an efficient architecture (as the state-action space is quite large). 
