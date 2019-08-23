#!/usr/bin/env bash

# Make sure we exit if there is a failure
set -e


function usage() {
    echo "Usage: $0 [--disable-inlining] [--ipdse] [--ai-dce] [--devirt VAL1] [--inter-spec VAL2] [--intra-spec VAL2] [--link dynamic|static] [--help]"
    echo "       VAL1=none|dsa|cha_dsa"    
    echo "       VAL2=none|aggressive|nonrec-aggressive"
}
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
        -g|--grpc)
            OPT_OPTIONS="${OPT_OPTIONS} --grpc"
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



set -- "${POSITIONAL[@]}" # restore positional parameters

#check that the required dependencies are built
declare -a bitcode=("bzip2.bc" "blocksort.o.bc" "huffman.o.bc" "crctable.o.bc" "randtable.o.bc" "compress.o.bc" "decompress.o.bc" "bzlib.o.bc")

for bc in "${bitcode[@]}"
do
    if [ -a  "$bc" ]
    then
        echo "Found $bc"
    else
        echo "Error: $bc not found. Try \"make\"."
        exit 1
    fi
done

DATABASE=${PWD}/slash/$PREFIX/
SLASH_OPTS="--inter-spec-policy=${INTER_SPEC} --intra-spec-policy=${INTRA_SPEC} --devirt=${DEVIRT} --no-strip --stats $OPT_OPTIONS --database=${DATABASE} --epsilon=$EPSILON"

# OCCAM with program and libraries dynamically linked
function dynamic_link() {

    export OCCAM_LOGLEVEL=INFO
    export OCCAM_LOGFILE=${PWD}/slash/$PREFIX/occam.log

    rm -rf $DATABASE
    mkdir -p $DATABASE
    echo "============================================================"
    echo "Running httpd with dynamic libraries "
    echo "slash options ${SLASH_OPTS}"
    echo "============================================================"
    slash ${SLASH_OPTS} --work-dir=$DATABASE \
	  --amalgamate=$DATABASE/amalgamation.bc \
    --epsilon=-1 \
	  bzip2.json
    status=$?
    if [ $status -ne 0 ]
    then
	echo "Something failed while running slash"
	exit 1
    fi
    ROPgadget --binary $DATABASE/bzip2 > $DATABASE/rop_stats.txt
    #cp ./slash_specialized/bzip2_slashed .
}

dynamic_link
