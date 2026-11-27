"""skia's bundled fontconfig is older than the system's and complains about its config
("Fontconfig error: ... invalid attribute 'xsi:nil'"), harmlessly, from native code writing to
fd 2, once: when skia's font manager starts. Its default config path is compiled in, so it can't
be pointed elsewhere; instead the font manager is started early with fd 2 sent to a file."""

import os
import sys
import tempfile


def start_fontconfig_quietly() -> None:
    """Start skia's font manager now, dropping fontconfig's complaints and passing on whatever
    else was written to fd 2 meanwhile. Call once, early, before other threads print."""
    import skia

    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            skia.FontMgr.RefDefault().countFamilies()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        for line in tmp.read().decode(errors="replace").splitlines():
            if not line.startswith("Fontconfig "):
                print(line, file=sys.stderr)
