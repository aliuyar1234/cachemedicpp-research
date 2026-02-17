# Security Notes - CacheMedic++ Research Repository

CacheMedic++ research intersects with security because KV caches can be corrupted or manipulated.

## Scope
- The project is a **defensive robustness** effort: stabilizing inference under cache corruption/faults/attacks.
- The repository includes corruption operators to **evaluate** robustness; these are minimal and transparent.

## Responsible usage
- Do not package exploit automation beyond the corruption functions needed to reproduce evaluation protocols.
- When publishing, describe corruption families and parameters, and avoid operational details that materially increase attack capability.

## Threat surfaces covered
- Integrity: transient faults (noise, zeroing, quantization), sparse sign/jump perturbations (bitflip-ish), structured overwrite.
- Privacy: multi-tenant KV cache sharing is out-of-scope for mitigation, but motivates the broader importance of cache safety.

## Out-of-scope
- Cache eviction, compression, clustering, or any memory reduction framing.
- Multi-tenant serving policy design.

