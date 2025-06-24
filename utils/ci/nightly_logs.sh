#!/usr/bin/env bash
# name: Tell Plato what logs are current
# description: Upload log IDs

# inputs:
#   deploy_key:
#     description: deploy key
#     required: true
#   run_id:
#     description: GitHub action run ID
#     required: true
#   job_id:
#     description: GitHub job ID
#     required: true
#   platform:
#     description: Platform for which ChimeraX is being built
#     required: true
#   build_type:
#     description: The type of ChimeraX build
#     required: true

DEPLOY_KEY=$1
RUN_ID=$2
JOB_ID=$3
PLATFORM=$4
BUILD_TYPE=$5

curl -v \
  -F "key=${DEPLOY_KEY}" \
  -F "run_id=${RUN_ID}" \
  -F "job_id=${JOB_ID}" \
  -F "platform=${PLATFORM}" \
  -F "build_type=${BUILD_TYPE}" \
  https://preview.rbvi.ucsf.edu/chimerax/cgi-bin/register_log_ids.py
