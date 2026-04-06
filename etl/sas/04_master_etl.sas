/*=============================================================================
  Program  : 04_master_etl.sas
  Project  : RAG ETL Pipeline
  Purpose  : Master controller – runs the full SAS ETL in sequence with
             error trapping, timing, and a final run summary.
  Author   : RAG ETL Team
  Created  : 2026-04-06

  Usage    : sas 04_master_etl.sas
             sas 04_master_etl.sas -set ENV prod
=============================================================================*/

options symbolgen mlogic mprint nosyntaxcheck;

/* ── Global parameters ──────────────────────────────────────────────────── */
%global ENV START_DT RUN_ID;
%let ENV      = %sysfunc(coalescec(&ENV., dev));
%let START_DT = %sysfunc(datetime(), datetime20.);
%let RUN_ID   = %sysfunc(putn(%sysfunc(today()), yymmddn8.))_%sysfunc(compress(%sysfunc(time(), time8.),':'));

%put NOTE: ============================================================;
%put NOTE:  RAG ETL Pipeline – Master Run;
%put NOTE:  ENV    = &ENV.;
%put NOTE:  RUN_ID = &RUN_ID.;
%put NOTE:  Start  = &START_DT.;
%put NOTE: ============================================================;

%let base = /home/user/RAG_ETL;
%include "&base./etl/sas/macros/etl_macros.sas";


/* ── Helper: run a step and capture its return code ─────────────────────── */
%macro run_step(step_label, step_file);
    %log_msg(--- BEGIN &step_label. ---);
    %let _step_start = %sysfunc(datetime());

    %include "&step_file.";

    %let _step_end  = %sysfunc(datetime());
    %let _step_secs = %sysevalf(&_step_end. - &_step_start.);
    %log_msg(--- END &step_label. – elapsed=%sysfunc(putn(&_step_secs., 8.1))s – SYSERR=&syserr. ---);

    %if &syserr. > 4 %then %do;
        %put ERROR: Step "&step_label." failed (SYSERR=&syserr.). Aborting pipeline.;
        %abort cancel;
    %end;
%mend run_step;


/* ── Execute pipeline stages ────────────────────────────────────────────── */
%run_step(01_Extract,   &base./etl/sas/01_extract.sas);
%run_step(02_Transform, &base./etl/sas/02_transform.sas);
%run_step(03_Load,      &base./etl/sas/03_load.sas);


/* ── Final summary ──────────────────────────────────────────────────────── */
%row_count(WORK.customers_clean,       n_cust);
%row_count(WORK.transactions_enriched, n_txn);
%row_count(WORK.customer_summary,      n_summary);

%put NOTE: ============================================================;
%put NOTE:  PIPELINE COMPLETE – &RUN_ID.;
%put NOTE:  customers_clean       = &n_cust. rows;
%put NOTE:  transactions_enriched = &n_txn.  rows;
%put NOTE:  customer_summary      = &n_summary. rows;
%put NOTE:  End = %sysfunc(datetime(), datetime20.);
%put NOTE: ============================================================;
