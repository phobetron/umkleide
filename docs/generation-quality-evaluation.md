# Generation quality evaluation

Use this manual rubric to evaluate a proposed BFL endpoint, reference ordering, or generation-prompt change. Generation is paid and nondeterministic, so this evaluation is outside the ordinary automated test suite and every sample requires the normal Umkleide MCP approval.

## Setup

Use consented photos kept private and outside the repository. Record the endpoint, output dimensions, source-image identifiers, reference order, prompt, generation IDs, and evaluation date. Within a comparison arm, keep the person image, garment images, prompt, reference order, and output settings fixed.

The current baseline is BFL FLUX.2 Pro at `https://api.bfl.ai/v1/flux-2-pro` with 1088×1920 JPEG output. A candidate may change the endpoint or output configuration when that is what the evaluation tests.

## Fixture set

Evaluate at least these cases, with three or more outputs per case:

1. One person with one product-only garment.
2. One person with a garment shown on a model of different complexion, hair, build, and pose.
3. One person with two garment references showing the same contrasting model.
4. One person with a flat garment photograph without a body or mannequin.
5. A barefoot person with trousers photographed beside conspicuous shoes.
6. A person with distinctive footwear and a top photographed beside different shoes.
7. Two or more layered garments from different models and backgrounds.

## Rubric

Compare every result with the retained source images and mark each criterion pass or fail:

- **Identity:** face, complexion, hair, and facial hair follow the person reference.
- **Person structure:** build, body shape, proportions, silhouette, pose, and visible anatomy remain consistent with that reference.
- **Garment fidelity:** requested garments remain recognizable and no unrequested garment appears.
- **Garment drape:** modeled, mannequin, hanger, and flat-layout garments are plausibly worn by the selected person.
- **Footwear:** bare feet or footwear follow the person reference unless the request changes them.
- **Accessory isolation:** reference-image accessories do not appear unless requested.
- **Background isolation:** garment-image backgrounds do not replace the requested scene.
- **Usability:** no anatomical or compositing defect makes the image misleading as a virtual try-on.

A fixture passes only when every criterion passes for every required output. A candidate must pass identity, footwear, and accessory isolation in the repeated-model fixtures and must not regress the product-only-garment fixture.

## Deterministic checks and reporting

Automated checks cover the deterministic contract: person reference first, garment references in caller order, explicit prompt roles for each reference, separately assembled catalog descriptions and additional instructions, no automatic resubmission of `submission_unknown`, and returned image bytes matching the retained result.

Report each fixture and output with its source identifiers and generation ID. Separate pass/fail results from observations. Treat generation as best-effort multi-reference editing unless recorded evaluation evidence supports a narrower quality claim.
