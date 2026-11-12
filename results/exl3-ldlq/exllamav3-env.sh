# Sourced by the GB10 jobs that run exllamav3 bd1f519: checks the vendored source and its prebuilt extension by
# sha256, unpacks both into a fresh folder and puts them, this folder and the pilot venv's packages on PYTHONPATH.
# Needs $pilot, the Drama pilot folder.
exl3=$PWD/results/exl3-ldlq
sha256sum -c <<EOF
15cfc9a4984d04c7d40acaa98ea9cbe55b4b6cce6c35bf2da6bca5e6ca42f833  $exl3/exllamav3-bd1f519.tar.gz
f2d768df28b8521d0756ac3bbfb9bd19d002c9b39a43b7eb2fd7af70869db075  $exl3/exllamav3_ext-bd1f519-sm121.so.gz
EOF

work=$(mktemp -d)
tar -xzf "$exl3/exllamav3-bd1f519.tar.gz" -C "$work"
# the job image has no C++ compiler to build exllamav3's extension; build-exllamav3-ext.sh prebuilt it, and exllamav3
# imports an exllamav3_ext found on the path instead of compiling one
gunzip -c "$exl3/exllamav3_ext-bd1f519-sm121.so.gz" > "$work/exllamav3_ext.so"

nvidia-smi -L
# the venv's interpreter links into the host's uv store, outside the job; its packages need a Python 3.12
export PYTHONPATH=$work:$work/exllamav3-bd1f519:$exl3:$pilot/venv/lib/python3.12/site-packages
