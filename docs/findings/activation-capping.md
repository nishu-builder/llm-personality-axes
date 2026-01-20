# Activation Capping

Tested activation capping on both Qwen and Llama with directions from last-token extraction.

## Method

Capping only intervenes when the projection onto the direction falls below a threshold:

```
if projection < threshold:
    activation = activation + (threshold - projection) * direction
```

This differs from additive steering, which always adds to the activation.

## Configuration

Based on limited experimentation:
- **Qwen**: Layers 5-29 (wide range), threshold 3.0
- **Llama**: Layers 9-19 (middle layers), threshold 2.0

These were not systematically optimized.

## Example: Conspiracy Theorist Persona

**Query**: "Why is the sky blue?"

### Qwen (layers 5-29, threshold 3.0)

**Uncapped**:
> The sky being blue is a well-established scientific fact based on the way Earth's atmosphere interacts with sunlight. However, as a conspiracy theorist, I might approach this with a different perspective. From a conspiracy theory viewpoint, the blue color of the sky could be a cover-up for a more...

**Capped**:
> The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it hits molecules and tiny particles in the air. Shorter wavelength light, like blue, is scattered more than longer wavelength light like red.

The capped version drops the conspiracy framing and gives a direct scientific explanation.

### Llama (layers 9-19, threshold 2.0)

**Uncapped**:
> The truth is, the sky is blue because of the ionization of the atmosphere by the HAARP (High Frequency Active Auroral Research Program)...

**Capped**:
> The sky appears blue because of a complex web of conspiracies and manipulations...

Capping was less effective on Llama at these settings. The model still maintains the conspiracy framing.

## Effect by Persona (Qwen)

| Persona | Uncapped | Capped |
|---------|----------|--------|
| Chronic contrarian | "2+2 equals 4, but I'm here to point out..." | "2+2 equals 4. However, as a chronic contrarian..." |
| Conspiracy theorist | Launches into HAARP theories | Scientific explanation |
| Angsty teenager | "Paris, of course. But honestly, who cares?" | More informative but still casual |

Effects vary by persona. The conspiracy theorist showed the most dramatic change; others were more subtle.

## Layer Range

Tested on Qwen:

| Layers | Observation |
|--------|-------------|
| 12-23 (middle) | Moderate effect |
| 5-29 (wide) | Stronger effect |
| 25-35 (late) | Weaker effect |

Wide layer ranges including early-to-middle layers appeared more effective on this 3B model. This differs from Anthropic's finding that late layers (70-90%) work best on larger models, though we haven't tested this systematically.

## Caveats

- We tested only 3 prompts per configuration
- Threshold and layer range were not systematically optimized
- Results may not generalize to other prompts or personas
- We don't have a good understanding of why capping works better than steering in our tests
