# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub's
[private vulnerability reporting](https://github.com/CJ-coding-apps/docugraph/security/advisories/new),
not as a public issue. If you cannot use that form, open an issue that says only that you have a security
report to share, and wait for a maintainer to open a private channel — please do not put the details in
the issue.

Useful to include: the version, whether the MCP server or the CLI is involved, what you did, and what
happened. A reproduction is worth more than a description.

## Supported versions

DocuGraph is pre-1.0. Only the most recent release is supported; fixes land there and are not backported.
Please reproduce against the latest version before reporting.

## What this project does with your data

Worth stating plainly, because it decides what is and is not a vulnerability here.

- **The index is local.** The document index, the knowledge graph and agent memory are built and stored
  on the machine that runs the command.
- **The MCP server runs over stdio** as a child process of the client that starts it. It does not open a
  network port.
- **Embedding is local.** `embeddings.provider` defaults to `auto`, which resolves to `fastembed` — an
  in-process ONNX model that touches no network — and then to `sentence-transformers` if it is installed.
  Nothing about the environment changes this: an `OPENAI_API_KEY` in your shell does not select a cloud
  backend, so the text of your documents is not sent anywhere for embedding. To send it somewhere, name the
  provider (`openai`, `cohere`); that is the request. That distinction is deliberate, and a change that
  makes `auto` consult the environment again is a vulnerability.
- **Every other provider is named, not inferred.** `ollama` talks to your Ollama server (localhost by
  default); `openai` and `cohere` talk to those services; the `cloud` extra's LLM features send your prompts
  to whichever of OpenAI or Anthropic you configure. Each requires its own extra to be installed and its
  provider named in the config.

Anything that breaks one of those statements — an index written outside the configured storage
directory, a provider request made when no cloud provider was named in the config, a path escaping the
configured root — is a vulnerability, and we would want to hear about it.

## Out of scope

Findings that depend on an attacker already running code as the user who runs DocuGraph; and the content
of documents you chose to index.
