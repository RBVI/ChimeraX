import json
from pathlib import Path
import re

import numpy as np

from chimerax.core import colors


def test_all_colorbrewer_palettes_and_default_aliases():
    """Check every shipped ramp against its RGB data, including alias names."""
    source = json.loads(Path(colors.__file__).with_name('colorbrewer.json').read_text())
    actual = {}
    colors._read_colorbrewer(actual)
    expected_names = set()
    for scheme, ramps in source.items():
        for count, entries in ramps.items():
            if count == 'type':
                continue
            rgb = []
            for entry in entries:
                match = re.fullmatch(r'rgb\((\d+),(\d+),(\d+)\)', entry)
                assert match is not None
                rgb.append([int(c) / 255 for c in match.groups()] + [1])
            expected = np.asarray(rgb, dtype=np.float32)
            name = f'{scheme}-{count}'
            palette = actual[name.casefold()]
            expected_names.add(name.casefold())
            assert palette.name == name
            np.testing.assert_array_equal(palette.colors, expected)
            if count == ('6' if ramps['type'] == 'qual' else '5'):
                alias = actual[scheme.casefold()]
                expected_names.add(scheme.casefold())
                assert alias.name == scheme
                assert alias is not palette
                np.testing.assert_array_equal(alias.colors, expected)
                assert not np.shares_memory(alias.colors, palette.colors)
    assert set(actual) == expected_names


def test_reused_rgb_data_does_not_share_mutable_palette_arrays():
    palettes = {}
    colors._read_colorbrewer(palettes)
    # The light end of these grayscale ramps uses the same RGB text.
    short, long = palettes['greys-3'], palettes['greys-4']
    original = long.colors.copy()
    short.colors[:] = 0
    np.testing.assert_array_equal(long.colors, original)
