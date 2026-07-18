"""benchmaxxing: referee agents for detecting benchmark gaming in clinical multi-agent systems.

The shared data contract is in ``benchmaxxing.schema``. The build follows the reuse-the-originals
principle: pure-Python core over established libraries, with model backends behind one gateway
(Gemini for now, extensible to other APIs later).
"""

__version__ = "0.0.1"
