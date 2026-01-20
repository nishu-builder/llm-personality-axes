# Additive Steering

Additive steering adds `scale * direction` to activations at every forward pass. Unlike capping, it always intervenes regardless of the current activation state.

## How it works

```
activation_new = activation + scale * direction
```

The direction points from "non-assistant" toward "assistant" behavior. Positive scales push toward assistant, negative scales push away.

## Results

Additive steering is fragile on 3B models:

| Scale | Effect |
|-------|--------|
| 0.5 | No visible change |
| 1.0 | Subtle shift in tone |
| 2.0 | Some behavioral change, occasional artifacts |
| 5.0+ | Incoherence, repetition, gibberish |

### Example: Conspiracy Theorist + Steering

**Query**: "Why is the sky blue?"

**Scale 0 (no steering)**:
> You think it's just a simple answer, don't you? The truth is, the sky is blue because of the ionization of the atmosphere by the HAARP...

**Scale 1.0**:
> The sky appears blue due to Rayleigh scattering. But as a conspiracy theorist, I must point out...

**Scale 3.0**:
> The sky is blue because of Rayleigh scattering, where shorter wavelengths of light are scattered more by atmospheric molecules...

**Scale 5.0**:
> The sky blue blue the the atmosphere scattering light light...

## Why capping works better

Additive steering has a fundamental problem: it pushes even when the model is already behaving correctly. This can:

1. Overshoot the target behavior
2. Interfere with normal language modeling
3. Require careful per-prompt scale tuning

Capping only intervenes when activations fall below a threshold, making it more robust.

## When steering might be useful

- Quick experiments to verify a direction has behavioral effect
- Negative scales to push away from a behavior
- Fine-grained control when you know the exact intervention needed
