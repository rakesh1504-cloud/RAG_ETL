/*=============================================================================
  Program  : 01_extract.sas
  Project  : RAG ETL Pipeline
  Purpose  : Extract raw data from flat files and databases into SAS datasets.
  Author   : RAG ETL Team
  Created  : 2026-04-06

  Inputs   : - data/raw/customers.csv
             - data/raw/transactions.csv
             - ODBC DSN (optional – see libname section)

  Outputs  : WORK.customers
             WORK.transactions
             WORK.product_lookup
=============================================================================*/

options symbolgen mlogic mprint;
%let root = %sysfunc(pathname(work));   /* fallback; override with your path */
%let raw_data = /home/user/RAG_ETL/data/raw;
%let proc_data = /home/user/RAG_ETL/data/processed;


/*----------------------------------------------------------------------------
  1. LIBNAME Definitions
----------------------------------------------------------------------------*/

/* Processed output library */
libname PROCLIB "&proc_data.";

/* Optional: ODBC connection to a relational database
libname SRCDB odbc
    dsn      = "my_dsn"
    schema   = "dbo"
    readbuff = 5000;
*/


/*----------------------------------------------------------------------------
  2. Import Customers CSV
----------------------------------------------------------------------------*/

proc import
    datafile = "&raw_data./customers.csv"
    out      = WORK.customers_raw
    dbms     = csv
    replace;
    getnames  = yes;
    guessingrows = max;
run;

proc contents data=WORK.customers_raw varnum;
    title "RAW: customers";
run;


/*----------------------------------------------------------------------------
  3. Import Transactions CSV
----------------------------------------------------------------------------*/

proc import
    datafile = "&raw_data./transactions.csv"
    out      = WORK.transactions_raw
    dbms     = csv
    replace;
    getnames  = yes;
    guessingrows = max;
run;

proc contents data=WORK.transactions_raw varnum;
    title "RAW: transactions";
run;


/*----------------------------------------------------------------------------
  4. Import Product Lookup (pipe-delimited)
----------------------------------------------------------------------------*/

proc import
    datafile = "&raw_data./product_lookup.txt"
    out      = WORK.product_lookup
    dbms     = dlm
    replace;
    delimiter = '|';
    getnames  = yes;
run;


/*----------------------------------------------------------------------------
  5. Direct SQL Pass-Through (example – Oracle)
     Uncomment and update credentials as required.
----------------------------------------------------------------------------*/
/*
proc sql;
    connect to oracle (
        user     = "&db_user."
        password = "&db_pass."
        path     = "&db_tns."
    );

    create table WORK.orders_raw as
    select * from connection to oracle (
        select order_id,
               customer_id,
               order_date,
               total_amount,
               status
        from   orders
        where  order_date >= '01JAN2024'd
    );

    disconnect from oracle;
quit;
*/


/*----------------------------------------------------------------------------
  6. Quick row-count check
----------------------------------------------------------------------------*/

%macro check_counts(dslist);
    %let i = 1;
    %let ds = %scan(&dslist., &i., |);
    %do %while (%length(&ds.) > 0);
        proc sql noprint;
            select count(*) into :nobs from &ds.;
        quit;
        %put NOTE: &ds. has &nobs. rows.;
        %let i = %eval(&i. + 1);
        %let ds = %scan(&dslist., &i., |);
    %end;
%mend check_counts;

%check_counts(WORK.customers_raw|WORK.transactions_raw|WORK.product_lookup);
