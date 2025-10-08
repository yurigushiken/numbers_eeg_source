import importlib
import sys
from types import ModuleType


def _clear_plotting_from_sysmodules():
    for key in list(sys.modules.keys()):
        if key == 'code.utils.plotting' or key.startswith('code.utils.plotting'):
            sys.modules.pop(key, None)


def _install_dummy_mne(path_variant: str | None):
    """Install a minimal dummy mne package in sys.modules for import testing.

    path_variant:
      - 'channels' -> provides mne.channels.layout._find_topomap_coords
      - 'viz'      -> provides mne.viz._topomap._find_topomap_coords
      - None       -> provides neither (simulate missing helper)
    """
    # Wipe any real mne
    for k in list(sys.modules):
        if k == 'mne' or k.startswith('mne.'):
            sys.modules.pop(k, None)

    mne = ModuleType('mne')
    sys.modules['mne'] = mne

    if path_variant == 'channels':
        channels = ModuleType('mne.channels')
        layout = ModuleType('mne.channels.layout')
        def _find_topomap_coords(info, picks):
            return None  # we only test import path resolution
        layout._find_topomap_coords = _find_topomap_coords
        channels.layout = layout
        sys.modules['mne.channels'] = channels
        sys.modules['mne.channels.layout'] = layout

    elif path_variant == 'viz':
        viz = ModuleType('mne.viz')
        _topomap = ModuleType('mne.viz._topomap')
        def _find_topomap_coords(info, picks):
            return None
        _topomap._find_topomap_coords = _find_topomap_coords
        viz._topomap = _topomap
        sys.modules['mne.viz'] = viz
        sys.modules['mne.viz._topomap'] = _topomap


def test_plotting_imports_channels_variant():
    _clear_plotting_from_sysmodules()
    _install_dummy_mne('channels')
    mod = importlib.import_module('code.utils.plotting')
    assert hasattr(mod, '_find_topomap_coords')
    assert mod._find_topomap_coords is not None


def test_plotting_imports_viz_variant():
    _clear_plotting_from_sysmodules()
    _install_dummy_mne('viz')
    mod = importlib.import_module('code.utils.plotting')
    assert hasattr(mod, '_find_topomap_coords')
    assert mod._find_topomap_coords is not None


def test_plotting_imports_missing_variant():
    _clear_plotting_from_sysmodules()
    _install_dummy_mne(None)
    mod = importlib.import_module('code.utils.plotting')
    assert hasattr(mod, '_find_topomap_coords')
    assert mod._find_topomap_coords is None

