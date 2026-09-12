# LegalMind-RAG competitive architecture

## Product boundary

LegalMind-RAG separates three kinds of evidence instead of asking one language model to infer
everything from a prompt:

1. historical, de-identified cases for statistical learning and similar-case retrieval;
2. versioned official statutes for rules effective on an explicit `as_of_date`;
3. structured sentencing predictions with task-specific label masks and uncertainty.

The final evidence firewall blocks an unqualified answer when citations are not grounded, no case
evidence is available, the legal date is absent, or the statute source/version is not verified.
This is a deterministic control and can be tested independently of the generator.

## End-to-end contract

`LegalMindPipeline.analyze(fact, as_of_date=...)` performs:

1. charge candidate classification;
2. de-identified historical-case retrieval;
3. effective-statute retrieval filtered by source review status and date interval;
4. optional structured sentencing prediction;
5. citation validation and evidence-firewall review.

Every statute search hit retains its official source URL, promulgation/effective/expiry dates,
legal status, retrieval timestamp, checksum, and human-verification status. Publication dates of
datasets and judgment dates of historical cases remain distinct concepts.

## Safety semantics

- Missing `as_of_date` never means “current law”.
- Automatically downloaded but unreviewed law text is not presented as verified applicable law.
- An expired or not-yet-effective rule is excluded and recorded in `rejected_statutes`.
- Current statutes never rewrite historical-case labels.
- Missing evidence causes explicit degradation and manual review, not fabricated authority.
- The system is a research prototype and does not constitute legal advice.

## External inspiration and implementation independence

ChatLaw is used only as an architectural research reference for role separation, legal retrieval,
and output review. No ChatLaw source code, weights, or training corpus is incorporated. This avoids
mixing AGPL implementation obligations or unaudited third-party court documents into the project.

## Interview framing

The defensible project claim is not “a legal chatbot”. It is an auditable temporal legal RAG and
multi-task sentencing system: data provenance is versioned, label absence is modeled explicitly,
case and statute evidence are isolated, legal applicability is date-filtered, and every final claim
passes deterministic evidence controls before presentation.
