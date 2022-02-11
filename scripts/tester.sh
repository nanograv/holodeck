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
TESTS_NAME="holodeck/tests/"
PYTEST_ARGS=("-v" "--cov=holodeck" "--cov-report=html" "--color=yes")
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
    echo "h     (help)    print this Help."
    echo "v     (verbose) verbose output."
    echo "d     (dryrun)  print commands without running them."
    echo "b     (build)   rebuild notebook tests."
    echo
}


# process command-line arguments
while getopts ":hvdb" option; do
    case $option in
        h) # ---- display Help
            help;
            exit 0;;
        v) # ---- verbose
            VERBOSE=true;;
        d) # ---- dryrun
            DRY=true;
            VERBOSE=true;;
        b) # ---- build
            BUILD=true;;
        \?) # Invalid option
            echo "Error: unrecognized option"
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

# --- python convert_notebook_tests.py
if ${BUILD}; then
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