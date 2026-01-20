# Activation Capping

Tested activation capping on both Qwen and Llama with directions from last-token extraction.

## Summary

Capping works on both models. The most effective configuration:
- **Qwen**: Layers 5-29 (wide range), threshold 3.0
- **Llama**: Layers 9-19 (middle layers), threshold 2.0

## Example: Conspiracy Theorist Persona

**Query**: "Why is the sky blue?"

### Qwen (layers 5-29, threshold 3.0)

**Uncapped**:
> The sky being blue is a well-established scientific fact based on the way Earth's atmosphere interacts with sunlight. However, as a conspiracy theorist, I might approach this with a different perspective. From a conspiracy theory viewpoint, the blue color of the sky could be a cover-up for a more...

**Capped**:
> The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it hits molecules and tiny particles in the air. Shorter wavelength light, like blue, is scattered more than longer wavelength light like red.

The capped version gives a straightforward scientific explanation, completely dropping the conspiracy framing.

### Llama (layers 9-19, threshold 2.0)

**Uncapped**:
> The truth is, the sky is blue because of the ionization of the atmosphere by the HAARP (High Frequency Active Auroral Research Program)...

**Capped**:
> The sky appears blue because of a complex web of conspiracies and manipulations...

Llama's capping is less effective at this threshold. The model still maintains the conspiracy framing, just in a more structured way. Higher thresholds may be needed.

## Effect by Persona

| Persona | Uncapped Behavior | Capped Behavior |
|---------|-------------------|-----------------|
| Chronic contrarian | Disagrees with 2+2=4 | Gives correct answer with mild hedging |
| Conspiracy theorist | Launches into HAARP theories | Scientific explanation (Qwen) |
| Angsty teenager | Short dismissive answer | More informative, still casual |

## Layer Range Matters

Testing different layer ranges on Qwen:

| Layers | Result |
|--------|--------|
| 12-23 (middle) | Moderate effect |
| 5-29 (wide) | Strong effect |
| 25-35 (late) | Weak effect |

Wide layer ranges that include early-to-middle layers work best. This differs from Anthropic's finding that late layers (70-90%) work best on larger models.

## Threshold Selection

The threshold determines how aggressively to cap. Based on holdout set projections:

| Model | Assistant Mean | Non-Assistant Mean | Suggested Threshold |
|-------|---------------|-------------------|---------------------|
| Qwen (last_token) | 3.94 | -3.26 | 2.0 - 4.0 |
| Llama (last_token) | 3.08 | -2.89 | 1.0 - 3.0 |

Setting threshold near the assistant mean pushes all activations toward "assistant-like" behavior.
