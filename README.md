# LeS1mple: A Hierarchical JEPA World Model for Long-Horizon Multi-Agent Dynamics in Counter-Strike

LeS1mple is a world model project for Counter-Strike. It uses the [OpenCS2 Dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset), which provides thousands of hours of tick-aligned video, audio, input actions, and world state. The goal is to learn a predictive, latent state-action representation of gameplay that can be used for planning.

The name is a portmanteau of S1mple, arguably the greatest CS player ever, and [LeWorldModel](https://arxiv.org/abs/2603.19312), a Joint Embedding Predictive Architecture (JEPA) that leverages a stable, effective, and very simple training objective.

I chose CS because it is a challenging, partially observable, multi-agent environment that requires complex strategy, quick reaction times, and an efficient architecture, as the state-action space is quite large.

# Architecture design

One key design decision is the choice of state. What do we leave to the encoder and what do we leave to the predictor? Ideally the encoder should contain all relevant information that a real player can access/would want during a round. So this includes the current tick state and recollection of prior events like seeing a player. 

Consider a complex example. Imagine the agent spawns with the bomb on the terrorist side of Dust2 and witnesses the entire enemy team crossing Mid Doors towards the B site. The agent begins push A site to plant the bomb unobstructed by enemy players. When the player is about to peek Long, its latent should encode that it saw all the enemies go to B. During CEM planning, predicted future latents for the peek action should encode that no enemies were seen since this is the likely outcome.

I think the encoder should just encode current evidence. The predictor should be the larger, more complex model which encodes memory, dynamics, and counterfactual futures. It will take a compressed history of past events that somehow persists this relevant context.

## Encoder input representation and model architecture

Our encoder encodes current evidence. It simply takes a POV frame.

The encoder should be a fairly small VIT with a moderately sized embedding dimension.

## Predictor Input Representation

The predictor should either be a state space model or transformer and it should be large relative to the encoder.

# Software stack

I will use `timm` and `stable-worldmodel`.

# Random quick notes

Quick note thoughts:  Perhaps training an omniscent world model first that could then densely supervise the player model to avoid CEM planning? 
