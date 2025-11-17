#!/usr/bin/env bash

set -x

function setup_python_env() {
    python -V
    pip install virtualenv
    virtualenv env
    source env/bin/activate
    pip install -e '.[extras,docs,ufb_banding]'
}

function configure_git() {
    git config --global user.name "${SPATPY_GITLAB_USER_NAME}"
    git config --global user.email "${SPATPY_GITLAB_USER_EMAIL}"
    git remote set-url origin "https://${SPATPY_RW_TOKEN_NAME}:${K8S_SECRET_SPATPY_RW_TOKEN}@${CI_REPOSITORY_URL#*@}"
    git fetch
    git status
}

function build_docs() {
    pushd docs
    rm -rf _apidoc/ html && sphinx-build . html
    popd
}

function publish_pages() {
    build_docs
    cp -r docs/html public
}

function run_tests() {
    python3 -m pytest
}

function push_to_devpi() {
    devpi use https://devpi.dolby.net

    devpi login capture --password=${K8S_SECRET_CAPTURE_DEVPI_PASSWORD}

    DEVPI_INDEX_NAME=${CI_COMMIT_REF_SLUG:-$CI_COMMIT_BRANCH}

    devpi index -l | grep ${DEVPI_INDEX_NAME} || devpi index -c ${DEVPI_INDEX_NAME} volatile=True 
    devpi use ${DEVPI_INDEX_NAME}

    git checkout ${CI_COMMIT_BRANCH}
    python3 -m bumpver update -vv --allow-dirty

    flit build --setup-py

    pushd docs
    rm -rf _apidoc/ html && sphinx-build . html
    popd

    CURRENT_VERSION=`bumpver show -n -e | grep CURRENT_VERSION | sed 's/.*=//'`
    pushd docs/html
    zip -r ../../dist/spatpy-${CURRENT_VERSION}.doc.zip .
    popd
    devpi upload -v --debug --only-latest --from-dir dist
}

