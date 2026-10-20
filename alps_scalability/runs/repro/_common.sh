# Sourced by every repro script: project root (relative to this file), conda env, project-local IGM.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"   # override if conda lives elsewhere
# shellcheck disable=SC1090
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate "${IGM_ENV:-igm}"
IGM_DIR="$ROOT/igm"                                              # IGM v3.2.1
export PYTHONPATH="$IGM_DIR${PYTHONPATH:+:$PYTHONPATH}"           # project-local IGM first
export TF_USE_LEGACY_KERAS=0 TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
python -c "import igm,os,sys; p=os.path.dirname(igm.__file__); sys.exit(0 if p.startswith('$IGM_DIR/') else 'IGM not imported from $IGM_DIR: '+p)"
echo "[repro] IGM = $IGM_DIR ($(git -C "$IGM_DIR" describe --tags --always --long 2>/dev/null))"
