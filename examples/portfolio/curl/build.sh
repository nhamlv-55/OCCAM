#!/usr/bin/env bash

export OCCAM_LOGLEVEL=INFO
export OCCAM_LOGFILE=${PWD}/slash_simple/occam.log

#slash  --stats --work-dir=slash_simple --amalgamate=slash_simple/amalgamation.bc curl.manifest.previrt
#cp slash_simple/curl curl_slash_simple

slash  --stats --work-dir=slash_specialized --amalgamate=slash_specialized/amalgamation.bc curl.manifest.specialized
cp slash_specialized/curl curl_slash_specialized
