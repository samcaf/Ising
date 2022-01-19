# Ising

[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)



## Table of Contents

- [Path Variables](#pathvars)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)


## <a name="pathvars"></a> Path Variables
If you want to run code out of the Ising folder, don't forget to add Ising to your path variable.

For me, one way to do this is by adding the following to my .bash_profile:
```
_path_append() {
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

_path_append PATH /path/to/Ising
PYTHONPATH=$PATH
export PYTHONPATH
```
Also, in ising/utils/file_utils.py, change the variable isingPath to the relevant '/path/to/Ising' directory.


## <a name="maintainers"></a> Maintainers

[@samcaf](https://github.com/samcaf)

## <a name="contributing"></a> Contributing

PRs accepted.

Small note: If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## <a name="license"></a> License

MIT © 2021 Samuel Alipour-fard
