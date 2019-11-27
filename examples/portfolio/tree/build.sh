#!/usr/bin/env bash
# Make sure we exit if there is a failure

SLASH="/root/anaconda3/envs/py2/bin/slash" 
set -e

function usage() {
    echo "Usage: $0 [--disable-inlining] [--ipdse] [--ai-dce] [--devirt VAL1] [--inter-spec VAL2] [--intra-spec VAL2] [--help]"
    echo "       VAL1=none|dsa|cha_dsa"    
    echo "       VAL2=none|aggressive|nonrec-aggressive"
}
INTER_SPEC="none"
INTRA_SPEC="machine-learning"
DEVIRT="none"
OPT_OPTIONS=""
EPSILON="-10"
GRPC_CONN=""
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
        -grpc-conn|--grpc-conn)
            GRPC_CONN="$2"
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


#check that the require dependencies are built
declare -a bitcode=("tree.bc")

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

echo "Linking tree_from_bc"
clang++ tree.bc -o tree_from_bc

DATABASE=${PWD}/slash/$PREFIX/
export OCCAM_LOGLEVEL=INFO
export OCCAM_LOGFILE=${PWD}/slash/$PREFIX/occam.log

mkdir -p $DATABASE
# OCCAM
SLASH_OPTS="--inter-spec-policy=${INTER_SPEC} --intra-spec-policy=${INTRA_SPEC} --devirt=${DEVIRT} --no-strip --stats $OPT_OPTIONS --database=${DATABASE} --epsilon=$EPSILON --grpc-conn=$GRPC_CONN --work-dir=slash/$PREFIX"
echo "============================================================"
echo "Running with options ${SLASH_OPTS}"
echo "============================================================"
${SLASH} ${SLASH_OPTS} tree.manifest.constraints >slash/$PREFIX/stdout_log 2>slash/$PREFIX/stderr_log

#ROPgadget --binary slash/$PREFIX/tree > slash/$PREFIX/rop_stats.txt
python ${OCCAM_HOME}/razor/MLPolicy/GSA_util/GSA.py -r $PREFIX -f ${PWD} 
if [$GRPC_CONN == ""]
then
    exit 0
else
    python3 ${OCCAM_HOME}/razor/MLPolicy/notify.py -c $GRPC_CONN
fi
