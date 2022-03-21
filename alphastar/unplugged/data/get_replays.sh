#!/bin/bash
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Script for downloading Starcraft II replay datasets.
#

readonly DOWNLOADS_DIR=/tmp/starcraft2_replay_downloads
readonly S2CLIENT_GIT_REPO=https://github.com/Blizzard/s2client-proto.git
readonly S2CLIENT_TEMP_REPO_DIR=/tmp/s2client
readonly TEST_CSV_URL=https://storage.cloud.google.com/dm-starcraft-offline-datasets/offline-test.csv
readonly TRAIN_CSV_URL=https://storage.cloud.google.com/dm-starcraft-offline-datasets/offline-train.csv
readonly VENV_DIR=/tmp/alphastar_unplugged_get_data_venv

function get_user_config {
  echo
  echo 'In order to download replay datasets you must first:'
  echo
  echo '1. Create an account via the Blizzard Developer Portal (https://dev.battle.net)'
  echo '2. Create a new client for accessing the APIs'
  echo
  echo 'Once you have completed these steps, enter the client ID and secret '
  echo 'below:'
  echo

  read -p 'Client ID: ' client_id
  read -p 'Secret: ' secret

  echo
  echo 'Next enter the root directory where replay datasets will be downloaded.'
  echo 'This directory will be created if it does not exist already.'
  echo

  read -p 'Root directory for replays: ' replay_root_dir
  replay_root_dir="${replay_root_dir/#\~/$HOME}"  # Expand "~" to "${HOME}"
  mkdir -p replay_root_dir

  echo
  echo 'Specify a comma-separated list of Starcraft versions to download '
  echo 'replays for (from 4.8.2, 4.8.3, 4.8.4, 4.8.6, 4.9.0, 4.9.1, 4.9.2).'
  echo

  read -p 'Starcraft versions: ' versions_list
  IFS=', ' read -r -a versions <<< "${versions_list}"

  yes_no_prompt 'Download train dataset?' 'Y' && get_train='Y' || get_train='N'
  yes_no_prompt 'Download test dataset?' 'Y' && get_test='Y' || get_test='N'

  echo
  echo 'Settings:'
  echo
  echo "Client ID: ${client_id}"
  echo "Secret: ${secret}"
  echo "Root directory for saving replays: ${replay_root_dir}"
  echo "Starcraft versions to fetch: ${versions[@]}"
  echo "Get training data? ${get_train}"
  echo "Get test data? ${get_test}"
  echo

  yes_no_prompt 'Continue?' 'Y' || { echo "Aborted"; exit 1; }
}

function yes_no_prompt {
    local prompt default reply

    if [[ ${2:-} = 'Y' ]]; then
        prompt='Y/n'
        default='Y'
    elif [[ ${2:-} = 'N' ]]; then
        prompt='y/N'
        default='N'
    else
        prompt='y/n'
        default=''
    fi

    while true; do
        echo -n "$1 [$prompt] "
        read -r reply </dev/tty
        if [[ -z $reply ]]; then
            reply=$default
        fi
        case "$reply" in
            Y*|y*) return 0 ;;
            N*|n*) return 1 ;;
        esac
    done
}

# We can't simply `curl` the URLs since they require Google account sign-in.
# We could potentially use `gcloud`, but that requires installing the Cloud SDK
# and I'm not sure it's worth the hassle.
function prompt_user_to_download_csvs {
  train_csv_path="${replay_root_dir}/offline-train.csv"
  test_csv_path="${replay_root_dir}/offline-test.csv"
  if [[ ${get_train} == 'Y' ]]
  then
    echo "Download ${TRAIN_CSV_URL} and save it to ${train_csv_path}."
    read -p "Press <enter> when done"
    [[ -f "${train_csv_path}" ]] || { echo "${train_csv_path} not found" ; exit 1; }
  fi
  if [[ ${get_test} == 'Y' ]]
  then
    echo "Download ${TEST_CSV_URL} and save it to ${test_csv_path}."
    read -p "Press <enter> when done"
    [[ -f "${test_csv_path}" ]] || { echo "${test_csv_path} not found" ; exit 1; }
  fi
}

function clone_s2_client_repo {
  if [[ -d ${S2CLIENT_TEMP_REPO_DIR} ]]
  then
    echo "Using existing S2 client repo ${S2CLIENT_TEMP_REPO_DIR}"
  else
    echo "Cloning S2 client repo to ${S2CLIENT_TEMP_REPO_DIR}"
    git clone ${S2CLIENT_GIT_REPO} ${S2CLIENT_TEMP_REPO_DIR}
  fi
  pushd "${S2CLIENT_TEMP_REPO_DIR}/samples/replay-api"
}

function setup_python_env {
  echo 'Creating new Python virtualenv for running the download script'
  python -m pip install --user virtualenv
  python -m virtualenv $VENV_DIR
  source ${VENV_DIR}/bin/activate

  echo 'Installing dependencies'
  # `six` should be included in `requirements.txt` but isn't.
  python -m pip install -U pip six
  python -m pip install -r requirements.txt
}

function download_replays {
  local csv_path version train_or_test
  csv_path=$1
  version=$2
  train_or_test=$3
  echo "Downloading ${train_or_test} replays in ${csv_path} for version ${version}"
  python download_replays.py \
    --key=${client_id}  \
    --secret=${secret}  \
    --version=${version}  \
    --replays_dir="${replay_root_dir}/${version}/${train_or_test}"  \
    --download_dir="${DOWNLOADS_DIR}"  \
    --filter_version=delete \
    --replayset_csv="${csv_path}"
}

function main {
  set -e
  get_user_config
  prompt_user_to_download_csvs
  clone_s2_client_repo
  setup_python_env
  for version in "${versions[@]}"
  do
    if [[ ${get_train} == 'Y' ]]
      then download_replays ${train_csv_path} ${version} "train"
    fi
    if [[ ${get_test} == 'Y' ]]
      then download_replays ${test_csv_path} ${version} "test"
    fi
  done
  deactivate
  popd
  echo "Finished downloading replay data to ${replay_root_dir}."
}

main