#!/usr/bin/env bash
# Prereqs: Make sure OCCAM envirnment is setup and WLLVM is installed
# env vars should be setup

# set CC to clang. Dragonegg works too
export LLVM_COMPILER=clang
export WLLVM_OUTPUT=WARNING

## Run Occam


# set up manifests
 cat > lhttpd.manifest <<EOF
{ "main" : "lighttpd.bc"
, "binary"  : "lighttpd"
, "args"    : ["-D", "-m", "/", "-f", "myconf.conf"]
, "name"    : "lighttpd"
, "modules" : []
, "ldflags" : ["-flat_namespace", "-undefined", "suppress", "-ldl"] 
}

EOF
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

DATABASE=${PWD}/slash/$PREFIX/
export OCCAM_LOGLEVEL=INFO
export OCCAM_LOGFILE=${PWD}/slash/$PREFIX/occam.log

mkdir -p $DATABASE
# OCCAM
SLASH_OPTS="--inter-spec-policy=${INTER_SPEC} --intra-spec-policy=${INTRA_SPEC} --devirt=${DEVIRT} --no-strip --stats $OPT_OPTIONS --database=${DATABASE} --epsilon=$EPSILON"
echo "============================================================"
echo "Running with options ${SLASH_OPTS}"
echo "============================================================"
slash ${SLASH_OPTS} --work-dir=slash/$PREFIX lhttpd.manifest

python ${OCCAM_HOME}/razor/MLPolicy/GSA_util/GSA.py -r $PREFIX -f ${PWD} 
