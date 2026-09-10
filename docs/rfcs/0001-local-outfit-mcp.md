# RFC 0001: Local outfit-image MCP

Status: Current

## Purpose

Umkleide is a local MCP service for one person’s outfit-reference catalog and BFL outfit-image generation. It maintains people and clothing as durable local records, supports local MCP clients over Streamable HTTP or stdio, and keeps provider use explicit through approval.

## Architecture

Local clients use loopback Streamable HTTP or client-launched stdio with the same contract. Separate tasks can launch concurrent stdio server processes over the same data root. A cross-process media lock covers catalog mutations, quota admission, media writes, and automatic maintenance. Locks cover operations rather than server lifetimes. Per-job cross-process locks ensure that overlapping retrieval attempts reuse persisted status rather than polling or downloading the same provider job concurrently. Provider network calls run outside the media lock.

An HTTP server runs independently of its clients. A stdio server remains available after a tool response while its client harness keeps the process alive.

The data root contains SQLite metadata, normalized JPEG media, credentials, and server coordination state. The service is single-user. The reserved `me` person exists by default; other people can have names and aliases. Clothing belongs to an owner for organization and search, while any person may wear any item.

The server serializes catalog mutations and uses atomic local persistence. An all-in-one `create_outfit_photo` request can add a person photo, person description, and clothing references while making an outfit. After approval and catalog revalidation, its catalog entries and generation record commit as one local batch before provider submission.

## Inputs and media

Every photo input is a `PhotoSource`: base64 JPEG, PNG, or WebP data URI; a path relative to the private import directory; or a public HTTPS image URL. Accepted images are validated, orientation-normalized, and retained as JPEGs. The service preserves the source image content; it does not isolate garments, people, or backgrounds.

Person and clothing descriptions are part of the durable catalog. A person description captures build, proportions, measurements, sizes, and fit preferences. A clothing description captures construction, materials, silhouette, dimensions, drape, and how the garment fits any pictured model. On generation, Umkleide assembles the selected person description, each item description, and the caller’s additional outfit instructions. The caller uses the generation prompt for the desired outfit or scene without repeating catalog descriptions.

## MCP surface

Catalog tools create, list, retrieve, update, inspect, and approved-delete people and clothing. Generation tools create an outfit request, check its status, inspect a retained image, expose the exact local JPEG path for file-backed rendering when needed, list records, and approved-delete a generation. `get_diagnostics` reports locations, credential source, media capacity, and table counts without exposing credentials.

The resource URIs are:

- `umkleide://people/{person_id}/photo`
- `umkleide://person-photos/{person_photo_id}`
- `umkleide://clothing/{clothing_item_id}`
- `umkleide://clothing-photos/{clothing_photo_id}`
- `umkleide://generations/{generation_id}`
- `umkleide://generation-images/{generation_id}`

When `create_outfit_photo` or `get_generation` reports `ready`, that same `CallToolResult` contains the structured generation record and the exact retained JPEG as native image content. If the retained image cannot be read locally, the tool returns an error rather than a ready result without image content. `get_generation_image` is available for inspecting a retained result. `get_generation_image_file` returns the local absolute path to that same JPEG for file-backed rendering on the MCP server host.

## Credentials and provider boundary

`umkleide-mcp configure` stores a BFL key in the private application-data directory without accepting it as a command-line argument. On startup, a nonblank `BFL_API_KEY` environment value overrides and persists over the stored value; otherwise the stored value is used. With neither, catalog functions remain available and generation does not start.

`create_outfit_photo` accepts one to six garments and requires an MCP approval form before BFL receives photos, descriptions, and additional instructions. Before that form, it validates the nonblank prompt, item count, BFL credential, selected person and clothing references, owners of supplied clothing, all selected and supplied photo bytes, and media quota. It decodes and normalizes supplied images in memory without changing durable catalog state. The approval explains that BFL receives the person reference, garments in caller order, stored or supplied descriptions, and the additional instructions; it also explains that BFL use can incur a charge. Declining leaves no supplied catalog entries or generation record.

After acceptance, the service revalidates the prepared catalog snapshot. It atomically records supplied person or clothing entries and the generation before the paid provider submission. The service owns the complete approved commit, submit, and receipt-classification path; cancellation of the MCP request does not cancel it. A confirmed receipt creates `processing`; a rejected submission becomes `failed`; an uncertain submission becomes `submission_unknown`.

## Lifecycle and retention

Generation states are `submission_unknown`, `processing`, `ready`, and `failed`. `ready` means the result JPEG is retained locally. `create_outfit_photo` accepts strict integer `wait_seconds` from 0 through 60 and defaults to 60; it waits only after submission. `get_generation` defaults to `wait_seconds=0` for one immediate advancement attempt, while a positive integer through 60 waits for server-owned retrieval.

Each server process starts a background retriever when it has a provider. It snapshots all `processing` records at startup, advances them in batches of up to four, and pauses five seconds before its next sweep. Per-job locks prevent duplicated provider work across processes, and a durable minimum five-second interval limits each job's provider polling. Retrieval exists only while an HTTP server is running or a stdio client harness keeps its server process alive. It stops when all server processes exit and resumes at startup. Provider delivery expiration can limit recovery of an unfinished job.

During shutdown, the server closes submission intake, gives accepted submissions up to 35 seconds to finish, then cancels and drains retrieval and local writes before closing its provider connection.

`retrieval_unavailable` leaves a job `processing`; continue checking the same ID. `storage_unavailable` also leaves it `processing`, but local storage or permissions must be corrected before another wait. `provider_failed` and `result_unavailable` make a job `failed`. `submission_unknown`, `failed`, and `result_unavailable` are reported without automatic submission. When transport times out before a generation ID is returned, inspect generation history before retrying.

Maintenance runs automatically at startup and during ordinary catalog and generation work. It immediately removes superseded person and clothing photo versions once no generation references them, removes unused media references, and deletes `failed` and `submission_unknown` records older than 30 days. It never automatically deletes `ready` outputs, current catalog photos, or records that are `processing`.

Intentional deletion uses approved `delete_person`, `delete_clothing_item`, or `delete_generation`. `delete_generation` applies to finished and failed records, removing the record and retained media; retrying is safe when an operation is interrupted before completion. Unresolved and processing generations remain protected from direct deletion.
