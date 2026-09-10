"""Shared reference-description guidance for MCP input schemas."""

PERSON_DESCRIPTION_GUIDANCE = (
    "Thorough fit-relevant description. Examine the supplied person photo and infer detailed "
    "visible information about overall build and body shape; frame; shoulder width and slope; "
    "chest, waist, and hip proportions; torso, arm, and leg proportions; posture; and how "
    "clothing sits on the body. Include clothing sizes, height, and fit preferences when "
    "available. Automatically included in BFL prompts when this person is used for outfit "
    "generation."
)
CLOTHING_DESCRIPTION_GUIDANCE = (
    "Thorough fit-relevant clothing description. Examine the supplied item photo and infer "
    "detailed information about garment type, size, cut, silhouette, proportions, length, "
    "shoulder and sleeve shape, neckline or collar, rise or waist position, leg shape, ease "
    "through the chest, waist, and hips, fabric structure, weight, drape, and stretch, closures, "
    "intended fit, and layering behavior. When the item's photo shows it worn by a model, infer "
    "and include an exact description of how it fits the model, including where it is fitted, "
    "relaxed, oversized, cropped, long, taut, or loose and where hems and sleeves fall. "
    "Automatically included in BFL prompts when this item is used for outfit generation."
)
OUTFIT_PROMPT_GUIDANCE = (
    "Nonblank additional outfit-generation instructions sent to BFL. The selected person's "
    "description "
    "and every selected or new item's description are included automatically; do not repeat "
    "those descriptions here."
)
