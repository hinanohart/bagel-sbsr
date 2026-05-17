# Safety and Dual-Use Policy — bagel-sbsr (Unified Multimodal Generation)

## What this project is

bagel-sbsr is a research implementation of a unified multimodal generation architecture
combining Sparse-Balanced Soft Routing (SBSR) with a Mixture-of-Transformers (MoT) backbone.
It targets text-to-image, image-to-image, and mixed-modal generation at research scale.

## Potential misuse areas

Unified multimodal generation systems can be misused to produce:

- **NSFW content** including sexually explicit imagery, including imagery involving minors (CSAM).
- **Impersonation content** — photorealistic synthetic media depicting real individuals in
  contexts they did not consent to.
- **Political deepfakes** — fabricated video or images depicting public figures making
  statements or performing actions they did not.
- **Non-consensual intimate imagery (NCII)** — synthetic intimate depictions of real people.

These use cases are explicitly prohibited. See the Contribution Policy below.

## Safeguards in this codebase

| Safeguard | Evidence |
|-----------|----------|
| LAION-aesthetic v2 excluded from training data; safety-conscious dataset choice | `docs/DATA.md` |
| `.gitignore` excludes model checkpoint formats (`.safetensors`, `.bin`, `.pt`, `.ckpt`, `.gguf`) | `.gitignore` |
| Token bearer values scrubbed from exception messages | `scripts/download_bagel.py:84` |
| Smoke test suite exercises generation pipeline on CPU without committing to real outputs | `tests/` (pytest `smoke` marker) |

**Note on checkpoints**: this repository does not distribute model checkpoints. Users who
download checkpoints from third-party sources (HuggingFace Hub, etc.) are responsible for
verifying the licensing terms and any safety commitments attached to those weights.

## Upstream safety statement

bagel-sbsr builds on the [BAGEL](https://github.com/ByteDance-Seed/Bagel) architecture
released by ByteDance Seed. Users should review ByteDance's safety disclosures and any
applicable usage restrictions for the BAGEL-7B-MoT base weights before use.

Attribution and upstream license terms are documented in [`NOTICE`](NOTICE).

## Contribution policy

The following categories of contribution will be rejected:

- **Training data** containing or designed to generate CSAM, NCII, or non-consented
  biometric data.
- **Pipelines or scripts** designed to generate deepfakes of named real individuals.
- **Classifier bypasses** — modifications whose stated or apparent purpose is to defeat
  content filtering systems applied to the outputs of this model.
- **Checkpoint loaders** for weights known to be derived from filtered-out or restricted
  datasets (e.g., LAION-5B subsets removed for safety reasons) without equivalent filtering.

## Regulatory context

Generating synthetic intimate imagery of real people without consent is illegal in a growing
number of jurisdictions. Generating synthetic imagery of minors in a sexual context (CSAM)
is illegal under the laws of virtually every country. Users are solely responsible for
compliance with applicable law.

In the United States, the relevant statutes include 18 U.S.C. § 2256 (CSAM) and state-level
NCII laws. In the European Union, Directive 2011/93/EU and national implementations apply.
In Japan, the Act on Punishment of Activities Relating to Child Prostitution and Child
Pornography (法律第52号) applies to synthetic imagery.

## Responsible disclosure

If you discover a capability of this system that enables meaningful uplift toward the
prohibited categories above, please report it via GitHub's private security advisory
feature rather than opening a public issue.
