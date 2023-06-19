#!/bin/zsh
# =================================================================================================
# holodeck testing script
#
# 1) runs the script to convert target notebooks into tests
# 2) runs pytest on all holodeck tests
#
# =================================================================================================
set -e    # exit on error

CONVERTER_NAME="scripts/convert_notebook_tests.py"
PYTHON_BUILD_COMMAND=("setup.py" "build_ext" "-i")
TESTS_NAME="holodeck/tests/"
NOTEBOOK_TESTS_NAME="holodeck/tests/converted_notebooks/"
PYTEST_ARGS=("-v" "--cov=holodeck" "--cov-report=html:coverage" "--color=yes")
VERBOSE=false;
DRY=false;
BUILD=false;

# ---------

DIR_PACKAGE=$(dirname $(dirname $(realpath $0)) )
PATH_CONVERTER=${DIR_PACKAGE}/${CONVERTER_NAME}

# Display Help
function help()
{
    echo "Setup and run 'holodeck' tests."
    echo
    echo "Syntax: tester.sh [-h|v|d|b] [FILES/DIRS...]"
    echo
    echo "options:"
    echo "h   (help)    print this Help."
    echo "v   (verbose) verbose output."
    echo "d   (dryrun)  print commands without running them."
    echo "l   (list)    list collected tests without running them."
    echo "b   (build)   rebuild notebook tests."
    echo "s   (skip)    skip    notebook tests."
    echo "x   (exit)    exit on first failure."
    echo
}


# process command-line arguments
while getopts ":hvdbslx" option; do
    case $option in
        h) # ---- display Help
            help;
            exit 0;;
        v) # ---- verbose
            VERBOSE=true;;
        d) # ---- dryrun
            DRY=true;
            VERBOSE=true;;
        l) # ---- list
            PYTEST_ARGS+=("--collect-only");;
        b) # ---- build (notebook tests)
            BUILD=true;;
        s) # ---- skip (notebook tests)
            PYTEST_ARGS+=("--ignore=${NOTEBOOK_TESTS_NAME}");;
        x) # ---- exit (on first failure)
            PYTEST_ARGS+=("-x");;
        \?) # Invalid option
            echo "Error: unrecognized option: '${option}'"
            exit 2;;
    esac
done

# if arguments are provided, use those as the files to run tests on
shift $((OPTIND -1))
FILES="${@:1}";
if [ -n "${FILES}" ]; then
    PATH_TESTS=${FILES};
else
    PATH_TESTS="${DIR_PACKAGE}/${TESTS_NAME}";
fi

if ${VERBOSE}; then echo "==== holodeck tester.sh ===="; fi
if ${DRY}; then echo "DRYRUN"; fi
if ${VERBOSE}; then echo ""; fi

# --- Build

if ${BUILD}; then
    # build package
    if ${VERBOSE}; then
        echo "$(which python) ${PYTHON_BUILD_COMMAND}";
    fi
    if ! ${DRY}; then
        python "${PYTHON_BUILD_COMMAND[@]}";
    fi
    # build notebook tests
    if ${VERBOSE}; then
        echo "$(which python) ${PATH_CONVERTER}";
    fi
    if ! ${DRY}; then
        python $PATH_CONVERTER;
    fi
fi

# --- pytest -v --cov=holodeck --cov-report=xml --color=yes holodeck/tests/
if ${VERBOSE}; then
    echo "$(which pytest) ${PYTEST_ARGS} ${PATH_TESTS}";
fi
if ! ${DRY}; then
    pytest "${PYTEST_ARGS[@]}" ${PATH_TESTS};
fi

echo "";