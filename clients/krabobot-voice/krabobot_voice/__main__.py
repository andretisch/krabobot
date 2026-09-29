"""python -m krabobot_voice"""

from __future__ import annotations

import multiprocessing

from krabobot_voice.app import main

if __name__ == "__main__":
    # Required for frozen Windows builds (meeting capture uses spawn).
    multiprocessing.freeze_support()
    raise SystemExit(main())
