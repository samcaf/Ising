# ============================
# Path preparation:
# ============================
# Should be run from the root folder /Ising

# -------------------------
# PYTHONPATH:
# -------------------------
# Adding the Ising directory to the PYTHONPATH
# Must be used in the directory /path/to/Ising/
path_append() {
    if [ -n "$2" ]; then
        case ":$(eval "echo \$$1"):" in
            *":$2:"*) :;;
            *) eval "export $1=\${$1:+\"\$$1:\"}$2" ;;
        esac
    else
        case ":$PATH:" in
            *":$1:"*) :;;
            *) export PATH="${PATH:+"$PATH:"}$1" ;;
        esac
    fi
}

path_append PATH $PWD
path_append PYTHONPATH $PWD

# Preparing paths for figures and operators
sed -i "s|isingPath = .*|isingPath = '"$PWD"'|" ising/utils/file_utils.py
