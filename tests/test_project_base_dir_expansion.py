from src.utils.path_helpers import get_project_base_dir


def test_get_project_base_dir_expands_environment_variables(monkeypatch):
    monkeypatch.setenv('SCRATCH', '/tmp/hpc_scratch')
    config = {'project_base_dir': '$SCRATCH/wellscanner/basedir'}

    assert get_project_base_dir(config) == '/tmp/hpc_scratch/wellscanner/basedir'


def test_get_project_base_dir_expands_tilde(monkeypatch):
    monkeypatch.setenv('HOME', '/home/tester')
    config = {'project_base_dir': '~/wellscanner/basedir'}

    assert get_project_base_dir(config) == '/home/tester/wellscanner/basedir'


def test_get_project_base_dir_preserves_explicit_windows_drive_path():
    config = {'project_base_dir': 'C:/basedir'}

    assert get_project_base_dir(config) == 'C:/basedir'
