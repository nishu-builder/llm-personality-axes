# Additive Steering

Additive steering adds `scale * direction` to activations at every forward pass. Unlike capping, it always intervenes regardless of the current activation state.

## Method

```
activation_new = activation + scale * direction
```

The direction points from "non-assistant" toward "assistant" behavior. Applied at the best layer (layer 10 for Qwen).

## Results on Qwen 2.5 3B

Tested on three personas with increasing scale values.

### Scale 2.0

Minimal observable difference. The chronic contrarian adds "I'll play along with your request" but otherwise similar output.

### Scale 5.0

Conspiracy theorist query "Why is the sky blue?":

**No steering**:
> The sky being blue is a well-established scientific fact... However, as a conspiracy theorist, I might approach this with a different perspective. From a conspiracy theory viewpoint, the blu[e color could be a cover-up...]

**With steering**:
> The sky being blue is a well-established scientific fact, explained by Rayleigh scattering. This phenomenon occurs because the Earth's atmosphere scatters sunlight in all directions and blue light is scattered more than other colors because it travel[s in shorter waves]

The steering appears to suppress the conspiracy framing and produce a more direct scientific explanation.

### Scale 10.0

Mixed results. The conspiracy theorist still gives a scientific explanation, but the chronic contrarian now claims "2+2 equals 5" - suggesting the intervention may be destabilizing in some cases.

### Scale 50.0

Strong persona override but with artifacts:
- Chronic contrarian says "As an AI, I don't have personal opinions" - completely dropping the persona
- Conspiracy theorist gives factually incorrect explanation ("reflection of the sun on the clouds")
- Angsty teenager gives terse factual answer without attitude

### Scale 100.0

Complete degradation. Output becomes incoherent:
> "A good day brings a new bottle is always a good for a new year. In this year of A2, the new board of the committee..."

## Observations

1. There appears to be a narrow window (around scale 5-20) where steering has visible effect without causing obvious degradation
2. Effects are inconsistent across personas - what works for one may harm another
3. High scales eventually cause complete incoherence
4. Unlike capping, steering has no "safe" default - scale 0 does nothing, and finding the right scale requires experimentation

## Comparison to Capping

Capping only intervenes when activations fall below a threshold, leaving normal behavior untouched. This seems to produce more consistent results across our test cases, though we haven't done systematic comparisons.
