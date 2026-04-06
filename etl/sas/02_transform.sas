/*=============================================================================
  Program  : 02_transform.sas
  Project  : RAG ETL Pipeline
  Purpose  : Clean, standardise, enrich, and validate extracted datasets.
  Author   : RAG ETL Team
  Created  : 2026-04-06

  Inputs   : WORK.customers_raw
             WORK.transactions_raw
             WORK.product_lookup

  Outputs  : WORK.customers_clean
             WORK.transactions_clean
             WORK.customer_summary
=============================================================================*/

options symbolgen mlogic mprint;
%include "/home/user/RAG_ETL/etl/sas/macros/etl_macros.sas";


/*============================================================================
  SECTION 1 – CLEAN CUSTOMERS
============================================================================*/

data WORK.customers_clean;
    set WORK.customers_raw;

    /* ── 1.1  Standardise column names (already handled by PROC IMPORT)    */

    /* ── 1.2  Strip leading/trailing whitespace                            */
    first_name = strip(first_name);
    last_name  = strip(last_name);
    email      = lowcase(strip(email));
    phone      = compress(phone, , 'kd');  /* keep digits only              */

    /* ── 1.3  Parse and validate signup date                               */
    signup_dt = input(signup_date, anydtdte10.);
    format signup_dt date9.;

    if missing(signup_dt) then do;
        put "WARNING: Invalid signup_date for customer_id=" customer_id;
        delete;
    end;

    /* ── 1.4  Derive date parts                                            */
    signup_year    = year(signup_dt);
    signup_month   = month(signup_dt);
    signup_quarter = qtr(signup_dt);

    /* ── 1.5  Derive full name                                             */
    full_name = catx(' ', first_name, last_name);

    /* ── 1.6  Age validation                                               */
    if not missing(age) then do;
        if age < 0 or age > 120 then do;
            put "WARNING: Suspicious age=" age "for customer_id=" customer_id;
            age = .;
        end;
    end;

    /* ── 1.7  Age bucket                                                   */
    length age_group $20;
    select;
        when (age < 25)                age_group = 'Under 25';
        when (25 <= age < 35)          age_group = '25-34';
        when (35 <= age < 50)          age_group = '35-49';
        when (50 <= age < 65)          age_group = '50-64';
        when (age >= 65)               age_group = '65+';
        otherwise                      age_group = 'Unknown';
    end;

    /* ── 1.8  Drop raw/intermediate columns                                */
    drop signup_date;

    label
        customer_id  = "Customer ID"
        full_name    = "Full Name"
        email        = "Email Address"
        age          = "Age (years)"
        age_group    = "Age Bucket"
        signup_dt    = "Signup Date"
        signup_year  = "Signup Year"
        signup_month = "Signup Month (1-12)"
        signup_quarter = "Signup Quarter";
run;


/*============================================================================
  SECTION 2 – CLEAN TRANSACTIONS
============================================================================*/

data WORK.transactions_clean;
    set WORK.transactions_raw;

    /* ── 2.1  Parse transaction date                                       */
    txn_dt = input(transaction_date, anydtdte10.);
    format txn_dt datetime20.;
    if missing(txn_dt) then delete;

    /* ── 2.2  Numeric coercion – amount                                   */
    amount_num = input(compress(amount, '$,'), best12.);
    if amount_num < 0 then do;
        is_refund  = 1;
        amount_num = abs(amount_num);
    end;
    else is_refund = 0;

    /* ── 2.3  Status standardisation                                       */
    status = upcase(strip(status));
    if status not in ('COMPLETED','PENDING','FAILED','REFUNDED') then do;
        put "WARNING: Unknown status=" status "for txn_id=" transaction_id;
        status = 'UNKNOWN';
    end;

    /* ── 2.4  Enrich with product info (merge happens in PROC SQL below)  */

    drop transaction_date amount;
    rename amount_num      = amount
           transaction_id  = txn_id;

    label
        txn_id    = "Transaction ID"
        txn_dt    = "Transaction Timestamp"
        amount    = "Transaction Amount (USD)"
        status    = "Transaction Status"
        is_refund = "Refund Flag (1=Yes)";
run;

/* Deduplicate on txn_id, keep last occurrence */
proc sort data=WORK.transactions_clean nodupkey dupout=WORK.dup_txns;
    by txn_id;
run;

%put NOTE: Duplicates removed from transactions:;
proc sql; select count(*) as dup_count from WORK.dup_txns; quit;


/*============================================================================
  SECTION 3 – ENRICH: MERGE PRODUCT LOOKUP
============================================================================*/

proc sort data=WORK.transactions_clean; by product_id; run;
proc sort data=WORK.product_lookup;     by product_id; run;

data WORK.transactions_enriched;
    merge WORK.transactions_clean  (in=a)
          WORK.product_lookup      (in=b);
    by product_id;
    if a;                              /* keep all transactions              */
    if not b then product_name = 'UNKNOWN';
run;


/*============================================================================
  SECTION 4 – CUSTOMER SUMMARY (PROC SQL AGGREGATION)
============================================================================*/

proc sql;
    create table WORK.customer_summary as
    select
        c.customer_id,
        c.full_name,
        c.email,
        c.age_group,
        c.signup_year,
        count(t.txn_id)                as txn_count      label="Transaction Count",
        sum(t.amount)                  as total_spend     label="Total Spend (USD)"  format=dollar12.2,
        avg(t.amount)                  as avg_spend       label="Avg Spend (USD)"    format=dollar10.2,
        max(t.txn_dt)                  as last_txn_dt     label="Last Transaction"   format=date9.,
        sum(t.is_refund)               as refund_count    label="Refund Count",
        calculated total_spend /
            nullif(calculated txn_count, 0) as ltv        label="Lifetime Value"     format=dollar12.2
    from       WORK.customers_clean     as c
    left join  WORK.transactions_enriched as t
        on c.customer_id = t.customer_id
    group by
        c.customer_id,
        c.full_name,
        c.email,
        c.age_group,
        c.signup_year
    order by total_spend desc;
quit;


/*============================================================================
  SECTION 5 – DATA VALIDATION REPORT
============================================================================*/

%macro validate_dataset(ds, key_col, not_null_cols);
    %put NOTE: === Validating &ds. ===;

    /* Null check */
    %let cols = %sysfunc(tranwrd(&not_null_cols., %str( ), %str(,)));
    proc sql noprint;
        %let i = 1;
        %let col = %scan(&not_null_cols., &i., %str( ));
        %do %while (%length(&col.) > 0);
            select count(*) into :null_cnt from &ds. where missing(&col.);
            %if &null_cnt. > 0 %then
                %put WARNING: &ds..&col. has &null_cnt. null values.;
            %else
                %put NOTE: OK – &ds..&col. has no nulls.;
            %let i   = %eval(&i. + 1);
            %let col = %scan(&not_null_cols., &i., %str( ));
        %end;
    quit;

    /* Duplicate key check */
    proc sort data=&ds.(keep=&key_col.) out=_key_check nodupkey dupout=_key_dups; by &key_col.; run;
    proc sql noprint; select count(*) into :dup_cnt from _key_dups; quit;
    %if &dup_cnt. > 0 %then %put WARNING: &ds..&key_col. has &dup_cnt. duplicate keys.;
    %else                   %put NOTE: OK – &ds..&key_col. keys are unique.;
    proc datasets lib=work nolist; delete _key_check _key_dups; quit;
%mend validate_dataset;

%validate_dataset(WORK.customers_clean,    customer_id, customer_id email full_name);
%validate_dataset(WORK.transactions_clean, txn_id,      txn_id customer_id amount);
