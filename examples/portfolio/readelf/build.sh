#!/usr/bin/env bash

# Make sure we exit if there is a failure
set -e

function usage() {
    echo "Usage: build.sh [--inter-spec VAL] [--intra-spec VAL] [--help]"
    echo "       VAL=none|aggressive|nonrec-aggressive"
}

#default values
INTER_SPEC="none"
INTRA_SPEC="none"

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
    -intra-spec|--intra-spec)
	INTRA_SPEC="$2"
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

#check that the require dependencies are built
declare -a bitcode=("readelf.bc")

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

MANIFEST=readelf.manifest

export OCCAM_LOGLEVEL=INFO
export OCCAM_LOGFILE=${PWD}/slash/occam.log

rm -rf slash

SLASH_OPTS="--inter-spec-policy=${INTER_SPEC} --intra-spec-policy=${INTRA_SPEC}"
echo "============================================================"
echo "Running readelf without libraries"
echo "slash options ${SLASH_OPTS}"
echo "============================================================"
slash ${SLASH_OPTS} --no-strip --stats --devirt=dsa --work-dir=slash ${MANIFEST}
status=$?
if [ $status -eq 0 ]
then
    ## runbench needs _orig and _slashed versions
    cp slash/readelf readelf_slashed
    cp binutils/binutils/readelf readelf_orig
else
    echo "Something failed while running slash"
fi    
