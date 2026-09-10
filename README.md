# Umkleide

Umkleide is a local, single-user MCP server for keeping person and clothing references and generating outfit images with Black Forest Labs (BFL). It creates a default `me` profile, supports additional names and aliases, and lets any person wear any stored item.

Local clients share the same catalog through Streamable HTTP or client-managed stdio processes. Multiple stdio processes can run concurrently, including those launched by separate Codex tasks. Umkleide coordinates shared catalog changes and retrieves accepted generation jobs while a server process is running.

## Install

Use Python 3.10 or later and Git. `pipx` is the simplest user-wide installation:

```sh
pipx install git+https://github.com/phobetron/umkleide.git
```

If needed, run `pipx ensurepath` and restart the terminal. Equivalent alternatives are:

```sh
uv tool install git+https://github.com/phobetron/umkleide.git
uv tool update-shell
```

```sh
python -m pip install "umkleide @ git+https://github.com/phobetron/umkleide.git"
```

The final command belongs in an activated virtual environment or another Python installation you manage.

## Configure generation

A BFL API key is required to generate images; catalog work does not need one. On the computer that runs Umkleide, save it with:

```sh
umkleide-mcp configure
```

When `BFL_API_KEY` is nonblank in the command environment, `configure` uses it without prompting. Otherwise it prompts without echoing the key. The key is stored in `credentials.json` in Umkleide’s private OS-user application-data directory. It is never accepted through an MCP tool or command-line argument, and it is not written to SQLite, logs, or diagnostics.

To remove the stored key:

```sh
umkleide-mcp configure --clear
```

A nonblank `BFL_API_KEY` at server startup takes precedence over the stored key and is saved to `credentials.json`. A blank or absent variable uses the stored key. Restart the server after changing credentials. This makes `configure` especially useful for graphical clients that do not inherit a terminal environment reliably.

## Run and connect

Start the default local HTTP service:

```sh
umkleide-mcp
```

It serves Streamable HTTP at `http://127.0.0.1:8000/mcp`. The explicit form is:

```sh
umkleide-mcp --transport=streamable-http
```

Keep this process running while HTTP clients use the catalog. The server listens only on loopback.

For a client-managed local child process, configure stdio:

```sh
umkleide-mcp --transport=stdio
```

Stdio and HTTP use the same MCP contract, catalog, and credentials. The client starts a stdio process automatically; separate tasks can each start one without extra configuration. A stdio server remains available after a tool response while its client harness keeps the process alive.

### Codex app

For HTTP, add a server named `umkleide`, choose **Streamable HTTP**, and enter `http://127.0.0.1:8000/mcp` after the server starts.

For stdio, add a **STDIO** server with command `umkleide-mcp` and argument `--transport=stdio`, then restart the app when prompted. After restarting, open a task and ask the client to call `get_diagnostics`. A working result includes `data_root`, `bfl_api_key_source`, and `records`. See the [Codex MCP guide](https://learn.chatgpt.com/docs/extend/mcp).

### Codex CLI

```sh
# HTTP
codex mcp add umkleide --url http://127.0.0.1:8000/mcp

# stdio
codex mcp add umkleide -- umkleide-mcp --transport=stdio
```

Use one server entry for a data root. These definitions contain no secret; configure Umkleide first, or provide `BFL_API_KEY` through private process environment management.

### Claude clients

Claude Code supports the same choices:

```sh
# HTTP
claude mcp add --transport http --scope user umkleide http://127.0.0.1:8000/mcp

# stdio
claude mcp add --transport stdio --scope user umkleide -- umkleide-mcp --transport=stdio
```

For Claude Desktop, add this entry to the existing `mcpServers` object in `claude_desktop_config.json`, then restart the client:

```json
{
  "mcpServers": {
    "umkleide": {
      "command": "umkleide-mcp",
      "args": ["--transport=stdio"]
    }
  }
}
```

## Supply photos

Each photo input is one `PhotoSource` object:

```json
{"type": "data_uri", "data": "data:image/jpeg;base64,..."}
```

```json
{"type": "import", "path": "shirt.png"}
```

```json
{"type": "url", "url": "https://example.com/shirt.jpg"}
```

`data_uri` accepts base64 JPEG, PNG, or WebP. `import` paths are relative to the private import directory reported in MCP initialization instructions and diagnostics. `url` must be a public HTTPS image URL. Umkleide validates images, normalizes orientation, and stores accepted media as JPEGs.

## Make useful references

Reference images establish appearance; descriptions establish the measurements, proportions, construction, and fit that images cannot state reliably. Keep both detailed and consistent.

For a person, use a clear full-body image from head to toe. Close-fitting clothing makes body proportions easier to see. Describe build and frame; shoulder, chest, waist, hip, torso, arm, and leg proportions; posture; known measurements; usual sizes; and fit preferences. State known measurements directly. For example:

> 170 cm tall with a straight, medium build; shoulders and hips are about the same width; a proportionally long torso and slightly shorter legs; 91 cm bust, 76 cm waist, 94 cm hips, and 74 cm inseam; usually wears EU 38 and prefers a relaxed fit through the waist.

For clothing, use a clear composite when several views matter. Include product-only front and back views where possible, with close-ups or modeled views that reveal seams, closures, texture, proportions, and drape. Describe item type, labeled size, materials, construction, cut, silhouette, measurements, fabric behavior, and intended fit. For modeled references, say where the garment is fitted, relaxed, oversized, taut, cropped, or long, and where hems and sleeves fall. Product-only views help reduce accidental influence from a model’s identity, pose, footwear, accessories, or background.

## Use Umkleide

Create or update a person, then catalog clothing with its reference and description. Clothing ownership organizes lookup and does not limit who can wear an item.

> Add Susan, also known as my wife, using this full-body photo. She is 170 cm tall with a straight, medium build, a long torso, and slightly shorter legs. Her bust, waist, and hip measurements are 91, 76, and 94 cm; her inseam is 74 cm. She usually wears EU 38 and prefers a relaxed fit through the waist.

> Save this composite as my blue jacket. It is a size M, hip-length jacket in medium-weight, non-stretch denim with a boxy cut and dropped shoulders. Record its visible construction and how it fits the pictured model.

Use `create_outfit_photo` for an outfit. It accepts one to six items and can combine cataloged items with supplied clothing and a supplied person photo. The service automatically includes the selected person and item descriptions in the provider prompt, so use `prompt` for the requested styling, layering, pose, or background rather than repeating reference descriptions.

> Show Susan wearing my blue jacket and her black trousers. Leave the jacket open, tuck in the shirt, and keep her current pose and background.

Before presenting the approval form, Umkleide validates the prompt, item count, BFL credential, selected person and clothing references, owners of supplied clothing, photo bytes, and local-media quota. It prepares supplied images only in memory at this stage. The approval explains that BFL receives the person photo, garment photos, stored or supplied descriptions, and the additional instructions; BFL use can incur a charge. Declining leaves the catalog unchanged.

After approval, Umkleide revalidates the selected catalog records. It writes supplied catalog entries and the generation record as one local batch before submitting to BFL. The server owns that accepted submission through receipt classification, even when the MCP request is cancelled. A request that cannot determine whether BFL accepted the submission is `submission_unknown` and is never submitted automatically again.

`create_outfit_photo` waits after submission for a strict integer `wait_seconds` from 0 through 60; it defaults to 60. Its result always includes the structured generation record and includes the exact retained native JPEG in the same tool result when the status is `ready`. If the retained image cannot be read locally, the tool returns an error rather than a ready result without image content. If it remains `processing`, present its ID and call `get_generation` with the same ID and `wait_seconds=60`. A `ready` result should be presented from the image already returned in that response.

`get_generation` defaults to `wait_seconds=0`, which performs one immediate advancement attempt. A positive integer through 60 waits for the server-owned retriever. Retrieval snapshots all `processing` jobs at server startup, advances them in batches of up to four, then pauses five seconds before its next sweep. Per-job locks prevent duplicate provider work across processes, and each job has a minimum five-second provider-poll interval. Retrieval runs while an HTTP server is running or while a stdio client harness keeps its server process alive. If all server processes exit, retrieval stops and resumes when a server starts. Provider delivery expiration can limit recovery of an unfinished job.

For `retrieval_unavailable`, continue checking the same generation. For `storage_unavailable`, correct local storage or permissions before another wait. For `failed` with `provider_failed` or `result_unavailable`, and for `submission_unknown`, report the outcome and do not automatically submit another request. If transport times out before returning a generation ID, inspect generation history before retrying. Use `get_generation_image` to inspect a retained result when needed. `get_generation_image_file` returns the exact retained JPEG path for local-host rendering when the current client needs a file path.

Review every result for identity, proportions, garment construction and drape, footwear, accessories, and background before treating it as an accurate try-on.

## Local data and deletion

Umkleide stores SQLite metadata and normalized private JPEGs in its OS-user application-data directory. On POSIX, managed directories use mode `0700` and files use `0600`.

Routine maintenance runs at startup and during normal work. It immediately reclaims superseded person and clothing photo versions once no generation still references them, removes unused media references, and removes `failed` and `submission_unknown` generation records older than 30 days. It protects `processing` records, `ready` results, and current catalog photos. The server never automatically deletes a ready generation image or a current catalog photo.

Use the approved deletion tools only when you intend to remove data: `delete_person`, `delete_clothing_item`, and `delete_generation`. `delete_generation` intentionally removes a finished or failed generation record and image; if an operation cannot complete, it can be retried. Unresolved and processing generations remain protected from direct deletion. `get_diagnostics` is read-only and reports the resolved directories, BFL credential source, media usage, quota, and record counts without returning the key.

## Environment reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `BFL_API_KEY` | No | A nonblank startup value becomes active and is persisted to `credentials.json`; blank or absent uses the stored credential. |
| `UMKLEIDE_MEDIA_QUOTA_BYTES` | No | Positive media quota in bytes; defaults to 1 GiB. |

## Troubleshooting

If catalog calls work but generation cannot begin, call `get_diagnostics`. `bfl_api_key_source: none` means no active credential; run `umkleide-mcp configure` or adjust the server environment, then restart.

For a local-host renderer that requires a path, call `get_generation_image_file`. It returns the exact retained JPEG’s absolute local path without copying or modifying it.

For an `import` source, place the file in the import directory reported by diagnostics. If the client cannot write there, use a data URI or a public HTTPS image URL instead.
