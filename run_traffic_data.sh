#!/bin/bash
if ! command -v module &> /dev/null
then
    echo "\`module\` command not found. Not on HPC SLURM cluster. Exiting"
    exit 1
fi

export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
module load apptainer

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$PROJECT" || exit 1
ROOT="$(readlink -f .)"
RESULTS="$ROOT/traffic_results"
TERM_OUT="$RESULTS/terminal_outputs.txt"
DATA="$ROOT"/traffic_data
SHEET="$DATA/traffic_data_processed.parquet"

CATS='accident,alcohol,belts,chrg_sect_mtch,chrg_title,chrg_title_mtch,comm_license,comm_vehicle,fatal,hazmat,home_outstate,licensed_outstate,outcome,patrol_entity,pers_injury,prop_dmg,race,search_conducted,search_disposition,search_type,sex,stop_chrg_title,subagency,tech_arrest,unmarked_arrest,vehicle_color,vehicle_make,vehicle_type,violation_type,work_zone'
ORDS='year_of_stop,month_of_stop,vehicle_year,weeknum_of_stop,weekday_of_stop'

DROPS00=chrg_sect_mtch,chrg_title,chrg_title_mtch,search_conducted,search_disposition,search_type,stop_chrg_title,violation_type
DROPS01=chrg_sect_mtch,chrg_title,chrg_title_mtch,race,search_conducted,search_disposition,search_type,stop_chrg_title,violation_type
DROPS02=chrg_sect_mtch,chrg_title,chrg_title_mtch,search_conducted,search_disposition,search_type,sex,stop_chrg_title,violation_type
DROPS03=chrg_sect_mtch,chrg_title,chrg_title_mtch,race,search_conducted,search_disposition,search_type,sex,stop_chrg_title,violation_type
DROPS04=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,search_conducted,search_disposition,search_type,stop_chrg_title
DROPS05=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,race,search_conducted,search_disposition,search_type,stop_chrg_title
DROPS06=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,search_conducted,search_disposition,search_type,sex,stop_chrg_title
DROPS07=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,race,search_conducted,search_disposition,search_type,sex,stop_chrg_title
DROPS08=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,search_disposition,search_type,stop_chrg_title,violation_type
DROPS09=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,race,search_disposition,search_type,stop_chrg_title,violation_type
DROPS10=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,search_disposition,search_type,sex,stop_chrg_title,violation_type
DROPS11=chrg_sect_mtch,chrg_title,chrg_title_mtch,outcome,race,search_disposition,search_type,sex,stop_chrg_title,violation_type
DROPS12=chrg_sect_mtch,chrg_title,outcome,search_conducted,search_disposition,search_type,stop_chrg_title,violation_type
DROPS13=chrg_sect_mtch,chrg_title,outcome,race,search_conducted,search_disposition,search_type,stop_chrg_title,violation_type
DROPS14=chrg_sect_mtch,chrg_title,outcome,search_conducted,search_disposition,search_type,sex,stop_chrg_title,violation_type
DROPS15=chrg_sect_mtch,chrg_title,outcome,race,search_conducted,search_disposition,search_type,sex,stop_chrg_title,violation_type

OUT00="$RESULTS/outcome__race+sex"
OUT01="$RESULTS/outcome__nosex"
OUT02="$RESULTS/outcome__norace"
OUT03="$RESULTS/outcome__norace+nosex"
OUT04="$RESULTS/violation_type__race+sex"
OUT05="$RESULTS/violation_type__nosex"
OUT06="$RESULTS/violation_type__norace"
OUT07="$RESULTS/violation_type__norace+nosex"
OUT08="$RESULTS/search_conducted__race+sex"
OUT09="$RESULTS/search_conducted__nosex"
OUT10="$RESULTS/search_conducted__norace"
OUT11="$RESULTS/search_conducted__norace+nosex"
OUT12="$RESULTS/chrg_title_mtch__race+sex"
OUT13="$RESULTS/chrg_title_mtch__nosex"
OUT14="$RESULTS/chrg_title_mtch__norace"
OUT15="$RESULTS/chrg_title_mtch__norace+nosex"

TARGET00=outcome
TARGET01=outcome
TARGET02=outcome
TARGET03=outcome
TARGET04=violation_type
TARGET05=violation_type
TARGET06=violation_type
TARGET07=violation_type
TARGET08=search_conducted
TARGET09=search_conducted
TARGET10=search_conducted
TARGET11=search_conducted
TARGET12=chrg_title_mtch
TARGET13=chrg_title_mtch
TARGET14=chrg_title_mtch
TARGET15=chrg_title_mtch

mkdir -p "$RESULTS"

df_analyze() {
    mkdir -p "$1"
    apptainer run --home $(readlink -f .) --app python "$ROOT/df_analyze.sif" \
        "$ROOT/df-analyze/df-analyze.py" \
        --df "$SHEET" \
        --mode classify \
        --target "$2" \
        --ordinals $ORDS \
        --drops "$3" \
        --classifiers lgbm rf lr dummy \
        --norm robust \
        --nan median \
        --feat-select filter embed wrap \
        --embed-select lgbm linear \
        --wrapper-select step-up \
        --wrapper-model linear \
        --filter-method assoc pred \
        --filter-assoc-cont-classify mut_info \
        --filter-assoc-cat-classify mut_info \
        --filter-pred-classify f1 \
        --n-feat-filter 20 \
        --n-feat-wrapper 10 \
        --redundant-wrapper-selection \
        --redundant-threshold 0.01 \
        --htune-trials 100 \
        --htune-cls-metric f1 \
        --test-val-size 0.25 \
        --outdir "$1" 2>&1 | tee "$4"

}


df_analyze "$OUT00" "$TARGET00" "$DROPS00" "$OUT00"_outputs.txt
