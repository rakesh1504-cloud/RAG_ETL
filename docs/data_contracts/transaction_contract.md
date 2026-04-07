# Data Contract: TRANSACTIONS

**Version:** 1.4  
**Owner:** Data Engineering Team  
**Last reviewed:** 2026-04-07  
**Status:** Active  

---

## Purpose

This contract defines the expectations, guarantees, and obligations for the `DW.TRANSACTIONS` table.
This is an **append-only immutable fact table** – records are never updated or deleted after landing.

---

## Producer Obligations

The ETL pipeline (`etl/python/pipeline.py`) guarantees:

1. **Immutability** – Once a row is loaded, it is never modified or deleted.
2. **Completeness** – All Stripe charge events are captured within 30 minutes of occurrence.
3. **No zero amounts** – Records with `amount = 0` are rejected (Stripe does not emit zero-value events).
4. **Currency normalisation** – All amounts stored in USD after FX conversion.
5. **Referential integrity** – `customer_id` validated against `DW.CUSTOMERS`. Orphans go to `DW.TXN_ORPHAN_LOG`.
6. **Batch traceability** – Every row carries `etl_batch_id` (UUID) enabling batch-level rollback.
7. **Null enforcement** – `transaction_id`, `customer_id`, `amount`, `transaction_date`, `transaction_ts`, `status` are never NULL.

---

## Consumer Expectations

Teams consuming `DW.TRANSACTIONS` may rely on:

- Table is append-only; safe to cache `MAX(etl_load_ts)` as a high-watermark.
- `transaction_ts` is always UTC. No local timezone adjustment has been applied.
- Refunds appear as **separate rows** with negative `amount`, not updates to the original row.
- `status = failed` rows are present in the table (not filtered out). Exclude with `WHERE status != 'failed'` if needed.
- Micro-batch runs every 15 minutes; maximum latency from Stripe event to DW is 30 minutes.

---

## SLA

| Metric | Target |
|---|---|
| Event-to-DW latency | ≤ 30 minutes |
| Batch success rate | > 99.9% of micro-batches complete without error |
| Orphan resolution | Finance team resolves orphans within 1 business day |
| Schema change notice | 21 days notice (critical financial table) |

---

## Data Quality Checks

| Check | Logic | Action on failure |
|---|---|---|
| `no_null_transaction_id` | `%assert_no_nulls(TRANSACTIONS, transaction_id)` | Abort batch |
| `no_null_customer_id` | `%assert_no_nulls(TRANSACTIONS, customer_id)` | Abort batch |
| `valid_amount` | `amount != 0` AND `-999999.99 <= amount <= 999999.99` | Reject row |
| `referential_integrity_customer` | `customer_id IN (SELECT customer_id FROM DW.CUSTOMERS)` | Route to `TXN_ORPHAN_LOG` |
| `valid_status_values` | Must be in `{completed, pending, failed, refunded}` | Reject row |
| `no_future_transaction_date` | `transaction_date <= CURRENT_DATE` | Quarantine row |

---

## ETL Lineage

```
Stripe Webhook Events
        │
        ▼
Stripe Events API  →  etl/python/pipeline.py
                              │
                              ├─ Parse JSON payload
                              ├─ Amount: cents → dollars (÷ 100)
                              ├─ FX conversion via DW.FX_RATES
                              ├─ Derive transaction_date from transaction_ts
                              ├─ Validate referential integrity
                              ├─ Assign etl_batch_id (UUID)
                              └─ Append INSERT to DW.TRANSACTIONS
```

---

## Financial Compliance Notes

- This table is subject to SOX audit requirements. Do **not** delete rows; use `status = refunded` flag.
- All changes to ETL logic affecting this table require dual approval (Data Engineering + Finance Controller).
- Batch failure logs must be retained for 7 years per finance data retention policy.
- `etl_batch_id` allows full reconstruction of any historical load for audit purposes.

---

## Known Limitations

- FX rates are point-in-time at transaction date; historical FX corrections are not retroactively applied.
- `channel` is NULL for 100% of records before 2022-01-01.
- Stripe test-mode transactions may appear in non-production environments; filter with `etl_source != 'STRIPE_TEST'`.

---

## Change Log

| Version | Date | Change |
|---|---|---|
| 1.4 | 2026-04-07 | Added `etl_batch_id` for batch traceability; SOX notes |
| 1.3 | 2025-06-01 | Added `channel` column |
| 1.2 | 2025-01-15 | Migrated to Snowflake micro-batch architecture |
| 1.0 | 2023-06-01 | Initial contract |
