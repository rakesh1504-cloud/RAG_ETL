/*=============================================================================
  File     : etl_macros.sas
  Project  : RAG ETL Pipeline
  Purpose  : Reusable SAS macro utility library.
             %include this file at the top of any program.
  Author   : RAG ETL Team
=============================================================================*/


/*----------------------------------------------------------------------------
  %LOG_MSG – Timestamped log message
  Usage : %log_msg(Starting customer transform);
----------------------------------------------------------------------------*/
%macro log_msg(msg, level=NOTE);
    %put &level.: [%sysfunc(datetime(), datetime20.)] &msg.;
%mend log_msg;


/*----------------------------------------------------------------------------
  %DS_EXISTS – Check if a dataset exists; return 1/0
  Usage : %if %ds_exists(WORK.my_data) %then ...;
----------------------------------------------------------------------------*/
%macro ds_exists(ds);
    %sysfunc(exist(&ds.))
%mend ds_exists;


/*----------------------------------------------------------------------------
  %ROW_COUNT – Store row count of a dataset in a macro variable
  Usage : %row_count(WORK.customers, cnt);
          %put There are &cnt. customers;
----------------------------------------------------------------------------*/
%macro row_count(ds, mvar);
    %global &mvar.;
    proc sql noprint;
        select count(*) into :&mvar. trimmed from &ds.;
    quit;
%mend row_count;


/*----------------------------------------------------------------------------
  %DROP_IF_EXISTS – Drop a dataset only if it exists (no WARNING)
  Usage : %drop_if_exists(WORK.temp_table);
----------------------------------------------------------------------------*/
%macro drop_if_exists(ds);
    %if %ds_exists(&ds.) %then %do;
        proc datasets lib=%scan(&ds., 1, .) nolist;
            delete %scan(&ds., 2, .);
        quit;
        %log_msg(Dropped &ds.);
    %end;
%mend drop_if_exists;


/*----------------------------------------------------------------------------
  %ASSERT_NO_NULLS – Abort if a required column has nulls
  Usage : %assert_no_nulls(WORK.customers, customer_id);
----------------------------------------------------------------------------*/
%macro assert_no_nulls(ds, col);
    proc sql noprint;
        select count(*) into :_null_cnt trimmed
        from &ds. where missing(&col.);
    quit;
    %if &_null_cnt. > 0 %then %do;
        %put ERROR: &ds..&col. has &_null_cnt. null values. Aborting.;
        %abort cancel;
    %end;
    %else %log_msg(OK – &ds..&col. has no nulls.);
%mend assert_no_nulls;


/*----------------------------------------------------------------------------
  %STANDARDISE_DATES – Parse character dates to SAS date using anydtdte.
  Usage : %standardise_dates(WORK.transactions, txn_date txn_posted_date);
----------------------------------------------------------------------------*/
%macro standardise_dates(ds, col_list);
    data &ds.;
        set &ds.;
        %let i = 1;
        %let col = %scan(&col_list., &i., %str( ));
        %do %while (%length(&col.) > 0);
            &col. = input(&col., anydtdte10.);
            format &col. date9.;
            %let i   = %eval(&i. + 1);
            %let col = %scan(&col_list., &i., %str( ));
        %end;
    run;
%mend standardise_dates;


/*----------------------------------------------------------------------------
  %TRIM_ALL_CHAR – Strip whitespace from ALL character columns in a dataset
  Usage : %trim_all_char(WORK.customers);
----------------------------------------------------------------------------*/
%macro trim_all_char(ds);
    /* Get list of character variables */
    proc contents data=&ds. out=_meta(keep=name type) noprint; run;
    proc sql noprint;
        select name into :_char_cols separated by ' '
        from _meta where type = 2;
    quit;
    proc datasets lib=work nolist; delete _meta; quit;

    data &ds.;
        set &ds.;
        %let i = 1;
        %let col = %scan(&_char_cols., &i., %str( ));
        %do %while (%length(&col.) > 0);
            &col. = strip(&col.);
            %let i   = %eval(&i. + 1);
            %let col = %scan(&_char_cols., &i., %str( ));
        %end;
    run;
%mend trim_all_char;


/*----------------------------------------------------------------------------
  %EXPORT_CSV – Export a dataset to CSV with a timestamp in the filename
  Usage : %export_csv_ts(WORK.customer_summary, /output/dir, customer_summary);
----------------------------------------------------------------------------*/
%macro export_csv_ts(ds, outdir, basename);
    %let ts = %sysfunc(putn(%sysfunc(today()), yymmddn8.));
    proc export
        data    = &ds.
        outfile = "&outdir./&basename._&ts..csv"
        dbms    = csv
        replace;
    run;
    %log_msg(Exported &ds. → &outdir./&basename._&ts..csv);
%mend export_csv_ts;


/*----------------------------------------------------------------------------
  %FREQ_CHECK – Quick frequency table to the log (no output window)
  Usage : %freq_check(WORK.customers, age_group status);
----------------------------------------------------------------------------*/
%macro freq_check(ds, col_list);
    %let i = 1;
    %let col = %scan(&col_list., &i., %str( ));
    %do %while (%length(&col.) > 0);
        proc freq data=&ds. noprint;
            tables &col. / out=_freq_out(keep=&col. count percent);
        run;
        %put NOTE: Frequency for &ds..&col.:;
        data _null_;
            set _freq_out;
            put &col. "  n=" count "  pct=" percent;
        run;
        %let i   = %eval(&i. + 1);
        %let col = %scan(&col_list., &i., %str( ));
    %end;
    proc datasets lib=work nolist; delete _freq_out; quit;
%mend freq_check;
