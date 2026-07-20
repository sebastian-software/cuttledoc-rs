# ADR-0011: US English as the project language

## Status

Accepted

## Context

Cuttledoc is developed across Rust, native Apple adapters, Node bindings,
benchmark tooling, architecture documents, and GitHub coordination. Most
durable project material is already written in English, but the repository has
not stated which English convention is normative or how multilingual product
and benchmark content relates to the project language.

Without an explicit rule, documentation and collaboration can drift between
languages and spelling conventions. That makes terminology, search, review,
automation, and future contributor onboarding less consistent. At the same
time, transcription is inherently multilingual, so a project-language rule
must not alter evidence, localization, or user content.

## Decision

US English is the normative language for durable Cuttledoc project material.
Use American spelling and terminology consistently in:

- tracked documentation and architecture decisions;
- source-code identifiers and comments;
- public API names and reference documentation;
- CLI help, default diagnostics, and default source strings;
- test names and developer-facing assertion messages;
- commit messages and changelog or release material; and
- maintainer-authored GitHub issues, pull requests, reviews, and durable
  status summaries.

Conversation may happen in another language when useful. Any resulting
decision, requirement, or status that becomes part of the durable project
record is summarized in US English.

The project-language rule does not translate or normalize content whose
language is itself meaningful. Preserve the original language for:

- audio fixtures, reference transcripts, ASR outputs, and benchmark gold data;
- localization resources and tests;
- user-provided or externally supplied content;
- exact quotations, proper names, model identifiers, and upstream terminology;
  and
- examples whose purpose is to demonstrate non-English behavior.

Product localization remains a capability and is not constrained to English.
US English is the default authoring language for source material, not the only
language the product may accept or present.

Existing durable material is updated when touched; this decision does not
require a mechanical repository-wide rewrite. New material follows this ADR
immediately.

## Consequences

### Positive

- Project terminology and spelling remain consistent across implementation,
  documentation, releases, and GitHub.
- Search, review, and automated analysis operate over one canonical durable
  language.
- Multilingual benchmark evidence remains authentic and auditable.
- Contributors may still collaborate in another language without fragmenting
  the project record.

### Negative

- Non-native English contributors may need an additional editing pass.
- Existing material can temporarily contain inconsistencies until it is
  touched.
- Maintainers must distinguish project prose from language-sensitive evidence
  rather than applying automatic translation broadly.

## Validation

Reviews should treat non-US-English durable project prose as a normal
documentation issue while exempting the language-sensitive categories above.
No language-specific linter is required initially; repeated inconsistencies
may justify a lightweight terminology or prose check later.
