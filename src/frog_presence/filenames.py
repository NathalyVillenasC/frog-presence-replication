"""Deciding what a pond recording is from the name it was given.

The pond collection holds material of four different kinds — full cassette sides, the
ten-minute samples the article works with, short excerpts cut out of those samples, and
recordings made on a different device — and the only thing that distinguishes them is the
filename. This module holds that decision because the ten-minute samples it identifies are
the article's test material: a change to these patterns changes which files the replication
is built on.
"""

import re

# Five files lost the dot before "wav", so the dot is optional in ENDS_WAV rather than
# repaired beforehand. Trailing "\s*" absorbs a stray space seen in one filename.
ENDS_WAV = r"\s*\.?wav$"
CASSETTE = r"^cc\d{3}[AB]"
POZA = r"[-_ ]*(?:poza\s?[12][-_])?_?"

# Order matters only for readability here: no filename matches two of these patterns.
ROLE_PATTERNS = [
    ("side master", re.compile(CASSETTE + ENDS_WAV, re.I)),
    ("SP sub", re.compile(CASSETTE + POZA + r"sp\d[a-z](?:_\d+)?" + ENDS_WAV, re.I)),
    ("SP base", re.compile(CASSETTE + POZA + r"sp\d" + ENDS_WAV, re.I)),
    ("LS recorder", re.compile(r"^ls\d+" + ENDS_WAV, re.I)),
]


def classify_env_file(name: str) -> str:
    """Assign an environmental recording to a role based on its filename.

    Args:
        name (str): Bare filename.

    Returns:
        str: One of "side master", "SP base", "SP sub", "LS recorder", "unclassified".
    """
    for role, pattern in ROLE_PATTERNS:
        if pattern.match(name):
            return role
    return "unclassified"
