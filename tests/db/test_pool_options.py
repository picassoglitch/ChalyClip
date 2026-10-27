from chalybclip.db.connection import pool_options


def test_transaction_pooler_disables_statement_cache() -> None:
    dsn = "postgresql://u:p@aws-1-us-west-1.pooler.supabase.com:6543/postgres"
    assert pool_options(dsn) == {"statement_cache_size": 0}


def test_session_pooler_and_direct_keep_defaults() -> None:
    assert pool_options("postgresql://u:p@h.pooler.supabase.com:5432/postgres") == {}
    assert pool_options("postgresql://u:p@localhost/db") == {}
    assert pool_options("postgresql://u:p@h:notaport/db") == {}
