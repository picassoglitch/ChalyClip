-- Chalyb rebrand — rename the two brand-kit toggle columns that carried the
-- old product name. Migrations 013 and 050 keep the original column names
-- because they are already applied on existing databases; this migration
-- brings both fresh and existing databases to the new names. The Python
-- models (`BrandKitRow.show_chalybclip_credit` / `show_chalybclip_outro`)
-- and the dashboard form fields read the new names only.
--
-- Portable SQL: RENAME COLUMN is supported by SQLite >= 3.25 and Postgres.
ALTER TABLE brand_kits RENAME COLUMN show_nexoclip_credit TO show_chalybclip_credit;
ALTER TABLE brand_kits RENAME COLUMN show_nexoclip_outro TO show_chalybclip_outro;
