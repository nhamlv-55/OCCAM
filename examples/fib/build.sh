#!/usr/bin/env bash

#make the bitcode
#default values
INTER_SPEC="none"
INTRA_SPEC="machine-learning"
DEVIRT="none"
OPT_OPTIONS=""
EPSILON="-10"
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -inter-spec|--inter-spec)
            INTER_SPEC="$2"
            shift # past argument
            shift # past value
            ;;
        -folder|--folder)
            PREFIX="run$2"
            shift # past argument
            shift # past value
            ;;
        -epsilon|--epsilon)
            EPSILON="$2"
            shift
            shift
            ;;
        -intra-spec|--intra-spec)
            INTRA_SPEC="$2"
            shift # past argument
            shift # past value
            ;;
        -disable-inlining|--disable-inlining)
            OPT_OPTIONS="${OPT_OPTIONS} --disable-inlining"
            shift # past argument
            ;;
        -ipdse|--ipdse)
            OPT_OPTIONS="${OPT_OPTIONS} --ipdse"
            shift # past argument
            ;;
        -ai-dce|--ai-dce)
            OPT_OPTIONS="${OPT_OPTIONS} --ai-dce"
            shift # past argument
            ;;
        -devirt|--devirt)
            DEVIRT="$2"
            shift # past argument
            shift # past value
            ;;    
        -help|--help)
            usage
            exit 0
            ;;
        *)    # unknown option
            POSITIONAL+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done


export OCCAM_LOGLEVEL=INFO
export OCCAM_LOGFILE=${PWD}/slash/$PREFIX/occam.log

DATABASE=${PWD}/slash/$PREFIX/
rm -rf slash/$PREFIX
mkdir -p $DATABASE
#slash --no-strip --intra-spec-policy=aggressive --inter-spec-policy=none --stats --work-dir=slash/agg manifest 

SLASH_OPTS="--inter-spec-policy=${INTER_SPEC} --intra-spec-policy=${INTRA_SPEC} --devirt=${DEVIRT} --no-strip --stats $OPT_OPTIONS --database=${DATABASE} --epsilon=$EPSILON"
echo "============================================================"
echo "Running with options ${SLASH_OPTS}"
echo "============================================================"
slash ${SLASH_OPTS} --work-dir=slash/$PREFIX manifest

#debugging stuff below:
for bitcode in slash/$PREFIX/*.bc; do
    echo "$bitcode"
    ${LLVM_HOME}/bin/llvm-dis  "$bitcode" -o "$bitcode".hr.bc
done

exit 0
