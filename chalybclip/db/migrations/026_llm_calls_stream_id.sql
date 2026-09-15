-- Token T3 — attribute each LLM / provider cost row to its stream.
--
-- llm_calls already records every Claude call AND (since T4) every
-- AssemblyAI transcription, with tokens + cost_usd_micros + provider.
-- But there was no way to ask "what did THIS stream's run cost?" — the
-- rows weren't tagged with a stream.
--
-- stream_id is NULLABLE: non-pipeline LLM calls (ad-hoc, future API
-- surfaces) and pre-T3 rows legitimately have no stream. The pipeline
-- binds stream_id in structlog contextvars for the whole run, so the
-- router + transcribe service stamp it on every row they write without
-- threading it through any call site.

ALTER TABLE llm_calls ADD COLUMN stream_id TEXT;

CREATE INDEX idx_llm_calls_stream ON llm_calls (stream_id);
