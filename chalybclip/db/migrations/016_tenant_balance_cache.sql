-- Chalyb integration — slice NX.4.
--
-- Caches the user's cross-engine token balance on the tenant row so we can
-- render it in the dashboard nav (top-right chip) without making a network
-- call to Chalyb on every page request.
--
-- The cache is updated by the outbound usage reporter: after each LLM call
-- pushes its event to Chalyb, the response carries the fresh balance.
-- We write those four numbers here. Templates read directly — no fetch.
--
-- Why columns and not jsonb: SQLite's JSON1 functions are fine for reads but
-- the type ergonomics are nicer with plain columns when we know the shape
-- won't grow. Four ints + a timestamp covers everything.
--
-- `unlimited` is stored as INTEGER (0 / 1) since SQLite has no native bool;
-- the Pydantic model coerces it to a Python bool.
--
-- All four fields are nullable so newly-created tenants (no LLM calls yet)
-- render the chip with a "—" placeholder until the first reporter ping
-- populates them.

ALTER TABLE tenants ADD COLUMN cached_balance_remaining INTEGER;
ALTER TABLE tenants ADD COLUMN cached_balance_unlimited INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tenants ADD COLUMN cached_balance_monthly_used INTEGER;
ALTER TABLE tenants ADD COLUMN cached_balance_at TEXT;
