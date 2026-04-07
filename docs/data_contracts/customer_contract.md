# Data Contract: CUSTOMERS

**Version:** 2.1  
**Owner:** Data Engineering Team  
**Last reviewed:** 2026-04-07  
**Status:** Active  

---

## Purpose

This contract defines the expectations, guarantees, and obligations for the `DW.CUSTOMERS` table.
Any team consuming or producing data for this table must comply with the rules below.

---

## Producer Obligations

The ETL pipeline (`etl/python/pipeline.py` and `etl/sas/etl_customers.sas`) guarantees:

1. **Completeness** – Every active CRM account is present within 24 hours of creation.
2. **Uniqueness** – `customer_id` is unique. Duplicate detection runs before every load.
3. **Referential validity** – All foreign key relationships are validated pre-load.
4. **Null enforcement** – `customer_id`, `first_name`, `last_name`, `signup_date`, `status` are never NULL. Enforced by `%assert_no_nulls` SAS macro.
5. **Date standardisation** – `signup_date` is always ISO 8601 (YYYY-MM-DD). Enforced by `%standardise_dates` SAS macro.
6. **Status vocabulary** – `status` is always one of `{active, inactive, suspended}`. Any other value is rejected to `DW.ETL_REJECT_LOG`.
7. **Audit columns** – `etl_load_ts` and `etl_source` are always populated.

---

## Consumer Expectations

Teams consuming `DW.CUSTOMERS` may rely on:

- `customer_id` is stable and immutable; never reused after deactivation.
- Data is refreshed daily by 06:00 UTC (incremental) and fully on Sundays by 04:00 UTC.
- `age` may be NULL for ~3% of legacy records (pre-2020 CRM migration).
- `email` may be NULL for ~1% of legacy records (pre-2020 email mandate).
- `channel` is NULL for records before 2022-01-01.

---

## SLA

| Metric | Target |
|---|---|
| Daily freshness | Data available by 06:00 UTC |
| Error rate | < 0.1% of records rejected per run |
| Quarantine review | Finance team reviews quarantine within 2 business days |
| Schema change notice | 14 days notice to all registered consumers |

---

## Data Quality Checks

All checks run in the ETL pipeline before data is loaded to production.

| Check | Logic | Action on failure |
|---|---|---|
| `no_null_customer_id` | `%assert_no_nulls(CUSTOMERS, customer_id)` | Abort pipeline run |
| `valid_status_values` | Must be in `{active, inactive, suspended}` | Reject row to `ETL_REJECT_LOG` |
| `age_range_check` | `18 <= age <= 120` or NULL | Set age to NULL |
| `signup_date_not_future` | `signup_date <= today()` | Quarantine to `CUSTOMERS_QUARANTINE` |
| `email_format` | Regex RFC 5322 pattern when not NULL | Reject row to `ETL_REJECT_LOG` |

---

## ETL Lineage

```
CRM (Salesforce)  →  CSV extract  →  etl/python/pipeline.py
                                           │
                                           ├─ Transform (trim, dates, age)
                                           ├─ Validate (nulls, status, age range)
                                           └─ Load (SCD Type 1 merge)
                                                   │
                                              DW.CUSTOMERS
```

SAS processing additionally runs `etl/sas/etl_customers.sas` for:
- Date standardisation (`%standardise_dates`)
- Whitespace trimming (`%trim_all_char`)
- Null assertion (`%assert_no_nulls`)
- Row count validation (`%row_count`)

---

## Known Limitations

- Age is derived from CRM self-reported date of birth; not verified.
- Phone numbers are not normalised (stored in raw CRM format).
- Email uniqueness is not enforced at the table level; two customers may share an email (family accounts).

---

## Change Log

| Version | Date | Change |
|---|---|---|
| 2.1 | 2026-04-07 | Added `channel` column; added email format check |
| 2.0 | 2025-01-15 | Migrated to Snowflake; added `etl_batch_id` |
| 1.0 | 2023-06-01 | Initial contract |
