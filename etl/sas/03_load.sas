/*=============================================================================
  Program  : 03_load.sas
  Project  : RAG ETL Pipeline
  Purpose  : Write transformed SAS datasets to output targets:
               – CSV / Parquet flat files
               – SQLite via ODBC
               – Permanent SAS library
               – Summary report (PDF)
  Author   : RAG ETL Team
  Created  : 2026-04-06

  Inputs   : WORK.customers_clean
             WORK.transactions_enriched
             WORK.customer_summary

  Outputs  : data/processed/customers_clean.csv
             data/processed/transactions_enriched.csv
             data/processed/customer_summary.csv
             data/processed/etl_report.pdf
=============================================================================*/

options symbolgen mlogic mprint;

%let proc_data = /home/user/RAG_ETL/data/processed;

/* Ensure output directory exists (SAS on Linux) */
options noxwait noxsync;
x "mkdir -p &proc_data.";


/*============================================================================
  1 – EXPORT TO CSV
============================================================================*/

%macro export_csv(ds, outfile);
    proc export
        data    = &ds.
        outfile = "&proc_data./&outfile."
        dbms    = csv
        replace;
    run;
    %put NOTE: Exported &ds. → &proc_data./&outfile.;
%mend export_csv;

%export_csv(WORK.customers_clean,       customers_clean.csv);
%export_csv(WORK.transactions_enriched, transactions_enriched.csv);
%export_csv(WORK.customer_summary,      customer_summary.csv);


/*============================================================================
  2 – SAVE TO PERMANENT SAS LIBRARY
============================================================================*/

libname PROCLIB "&proc_data.";

data PROCLIB.customers_clean;       set WORK.customers_clean;       run;
data PROCLIB.transactions_enriched; set WORK.transactions_enriched; run;
data PROCLIB.customer_summary;      set WORK.customer_summary;      run;

%put NOTE: Datasets saved to permanent library PROCLIB (&proc_data.).;


/*============================================================================
  3 – WRITE TO DATABASE VIA ODBC (optional)
     Uncomment and configure your DSN.
============================================================================*/
/*
libname TGTDB odbc dsn="my_target_dsn" schema="etl_output";

data TGTDB.customers_clean / replace;
    set WORK.customers_clean;
run;

data TGTDB.customer_summary / replace;
    set WORK.customer_summary;
run;
*/


/*============================================================================
  4 – SUMMARY REPORT (PDF)
============================================================================*/

ods pdf file="&proc_data./etl_report.pdf"
        style=journal
        startpage=never;

title "RAG ETL Pipeline – Run Report";
title2 "Generated: %sysfunc(today(), worddate.)";

/* 4a – Customer overview */
title3 "Customer Demographics";
proc freq data=WORK.customers_clean;
    tables age_group signup_year / nocum;
run;

/* 4b – Transaction status breakdown */
title3 "Transaction Status Distribution";
proc freq data=WORK.transactions_enriched;
    tables status is_refund / nocum;
run;

/* 4c – Customer spend summary */
title3 "Top 20 Customers by Total Spend";
proc print data=WORK.customer_summary(obs=20) noobs label;
    var customer_id full_name age_group total_spend avg_spend txn_count last_txn_dt;
run;

/* 4d – Spend distribution */
title3 "Spend Distribution";
proc univariate data=WORK.customer_summary noprint;
    var total_spend;
    histogram total_spend / normal;
    inset mean std median / pos=ne;
run;

ods pdf close;
title;
%put NOTE: PDF report written to &proc_data./etl_report.pdf;


/*============================================================================
  5 – ROW COUNT AUDIT LOG
============================================================================*/

%macro audit_log(ds, step);
    proc sql noprint;
        select count(*) into :n from &ds.;
    quit;
    data _null_;
        file "&proc_data./audit_log.txt" mod;
        put "%sysfunc(datetime(), datetime20.) | &step. | &ds. | &n. rows";
    run;
    %put NOTE: AUDIT &step. &ds. = &n. rows;
%mend audit_log;

%audit_log(WORK.customers_clean,       LOAD);
%audit_log(WORK.transactions_enriched, LOAD);
%audit_log(WORK.customer_summary,      LOAD);
