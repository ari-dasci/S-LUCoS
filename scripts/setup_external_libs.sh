#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS_DIR="${ROOT_DIR}/libs"
TEMPLATE_DIR="${ROOT_DIR}/scripts/external_libs"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LIBS_DIR}"

log() {
  printf '[setup_external_libs] %s\n' "$*"
}

ensure_clean_submodule() {
  local repo_dir="$1"
  local name="$2"
  local dirty_entries=()
  local line
  local path

  if [[ "${ALLOW_DIRTY_EXTERNAL_LIBS:-0}" == "1" ]]; then
    return
  fi

  while IFS= read -r line; do
    path="${line:3}"
    case "${path}" in
      pyproject.toml|*.egg-info|*.egg-info/*|build|build/*|dist|dist/*|__pycache__|__pycache__/*)
        continue
        ;;
      checkpoints|checkpoints/*)
        if [[ "${name}" == "TabClustPFN" ]]; then
          continue
        fi
        ;;
      exp/__init__.py|model/__init__.py)
        if [[ "${name}" == "TabClustPFN" ]]; then
          continue
        fi
        ;;
    esac
    dirty_entries+=("${line}")
  done < <(git -C "${repo_dir}" status --porcelain --untracked-files=all)

  if [[ "${#dirty_entries[@]}" -gt 0 ]]; then
    cat >&2 <<EOF
${name} already exists at ${repo_dir} and has local changes.
Refusing to update the submodule because that could overwrite work.
Commit/stash those changes, remove the directory, or rerun with:
  ALLOW_DIRTY_EXTERNAL_LIBS=1 scripts/setup_external_libs.sh
EOF
    printf '%s\n' "${dirty_entries[@]}" >&2
    exit 1
  fi
}

is_initialized_submodule() {
  local repo_dir="$1"
  local top_level
  local repo_realpath
  local top_realpath

  [[ -d "${repo_dir}" ]] || return 1

  top_level="$(git -C "${repo_dir}" rev-parse --show-toplevel 2>/dev/null)" || return 1
  repo_realpath="$(cd "${repo_dir}" && pwd -P)"
  top_realpath="$(cd "${top_level}" && pwd -P)"

  [[ "${repo_realpath}" == "${top_realpath}" ]]
}

disable_filemode_tracking() {
  local repo_dir="$1"

  if ! git -C "${repo_dir}" config core.fileMode false 2>/dev/null; then
    log "Could not set core.fileMode=false for ${repo_dir}; continuing"
  fi
}

add_info_exclude() {
  local repo_dir="$1"
  shift
  local exclude_file
  local git_dir
  local pattern

  git_dir="$(git -C "${repo_dir}" rev-parse --absolute-git-dir)"
  exclude_file="${git_dir}/info/exclude"
  mkdir -p "$(dirname "${exclude_file}")"
  touch "${exclude_file}"

  for pattern in "$@"; do
    if ! grep -Fxq "${pattern}" "${exclude_file}"; then
      printf '%s\n' "${pattern}" >> "${exclude_file}"
    fi
  done
}

configure_generated_artifact_ignores() {
  add_info_exclude "${LIBS_DIR}/TabClustPFN" \
    "/pyproject.toml" \
    "/build/" \
    "/dist/" \
    "/tabclustpfn.egg-info/" \
    "/exp/__init__.py" \
    "/model/__init__.py" \
    "/checkpoints/"

  add_info_exclude "${LIBS_DIR}/RDSS" \
    "/pyproject.toml" \
    "/build/" \
    "/dist/" \
    "/rdss.egg-info/" \
    "/__pycache__/"

  add_info_exclude "${LIBS_DIR}/zcore" \
    "/pyproject.toml" \
    "/build/" \
    "/dist/" \
    "/zcore.egg-info/"
}

setup_submodule() {
  local name="$1"
  local path="libs/${name}"
  local repo_dir="${LIBS_DIR}/${name}"
  local expected_commit
  local needs_update=1

  expected_commit="$(git -C "${ROOT_DIR}" ls-files --stage "${path}" | awk '{print $2}')"
  if [[ -z "${expected_commit}" ]]; then
    printf '%s is not registered as a git submodule at %s\n' "${name}" "${path}" >&2
    exit 1
  fi

  if is_initialized_submodule "${repo_dir}"; then
    local current_commit
    disable_filemode_tracking "${repo_dir}"
    ensure_clean_submodule "${repo_dir}" "${name}"

    current_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
    if [[ "${current_commit}" == "${expected_commit}" ]]; then
      log "${name} submodule already at ${expected_commit}"
      needs_update=0
    fi
  fi

  if [[ "${needs_update}" == "1" ]]; then
    log "Updating ${name} submodule at ${expected_commit}"
    git -C "${ROOT_DIR}" submodule update --init --recursive "${path}"
  fi

  disable_filemode_tracking "${repo_dir}"

  local actual_commit
  actual_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    printf '%s is at %s, expected %s\n' "${name}" "${actual_commit}" "${expected_commit}" >&2
    exit 1
  fi
}

copy_template() {
  local source="$1"
  local target="$2"

  mkdir -p "$(dirname "${target}")"
  cp "${source}" "${target}"
}

apply_local_packaging() {
  copy_template "${TEMPLATE_DIR}/TabClustPFN.pyproject.toml" "${LIBS_DIR}/TabClustPFN/pyproject.toml"
  mkdir -p "${LIBS_DIR}/TabClustPFN/checkpoints"
  touch "${LIBS_DIR}/TabClustPFN/exp/__init__.py"
  touch "${LIBS_DIR}/TabClustPFN/model/__init__.py"

  copy_template "${TEMPLATE_DIR}/RDSS.pyproject.toml" "${LIBS_DIR}/RDSS/pyproject.toml"

  copy_template "${TEMPLATE_DIR}/zcore.pyproject.toml" "${LIBS_DIR}/zcore/pyproject.toml"
}

install_package() {
  local name="$1"
  log "Installing ${name}"
  "${PYTHON_BIN}" -m pip install --no-build-isolation "${LIBS_DIR}/${name}"
}

setup_submodule "RDSS"
setup_submodule "TabClustPFN"
setup_submodule "zcore"

configure_generated_artifact_ignores

log "Applying local packaging files"
apply_local_packaging

install_package "RDSS"
install_package "TabClustPFN"
install_package "zcore"

log "Done"
