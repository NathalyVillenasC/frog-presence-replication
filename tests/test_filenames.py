"""Tests for assigning pond-recording roles from filenames.

The filenames used here come directly from the collection, including malformed examples.
These tests document the cases the regular expressions are expected to handle and help detect
future changes that would alter how the replication classifies its test material.
"""

import pytest

from frog_presence.filenames import classify_env_file


@pytest.mark.parametrize(
    "name, role",
    [
        ("cc126A.wav", "side master"),
        ("cc130B.wav", "side master"),
        ("cc126A_SP1.wav", "SP base"),
        ("cc144B_poza2_SP2.wav", "SP base"),
        ("cc126A_SP1B.wav", "SP sub"),
        ("cc133A_SP1A_2.wav", "SP sub"),
        ("LS100001.wav", "LS recorder"),
    ],
)

def test_each_role_is_recognised(name, role):
    """Expected case: one well-formed name per role."""
    assert classify_env_file(name) == role


def test_a_filename_missing_the_dot_is_still_classified():
    """Edge case: five files in the collection lost the dot before the extension.

    They are ten-minute samples like any other and have to reach the same role, which is
    why the dot is optional in the pattern instead of modifiying the filenames beforehand.
    """
    assert classify_env_file("cc132A_SP1Bwav") == "SP sub"


def test_a_stray_space_before_the_extension_is_absorbed():
    """Edge case: one filename in the collection carries a space before ".wav"."""
    assert classify_env_file("cc144B_SP2B .wav") == "SP sub"


def test_the_sub_pattern_wins_over_the_base_pattern():
    """Edge case: an excerpt must not be mistaken for the sample it was cut from.

    A six-second excerpt and a ten-minute sample cannot be pooled in an experiment, and
    "cc126A_SP1B.wav" would satisfy the base pattern too if it were tried first.
    """
    assert classify_env_file("cc126A_SP1B.wav") == "SP sub"
    assert classify_env_file("cc126A_SP1.wav") == "SP base"


@pytest.mark.parametrize( "name", ["notes.txt", "cc126A.xmp", "cc126A.rtf", "random recording.wav", ""])

def test_anything_else_is_left_unclassified(name):
    """Failure case: an unrecognised name is labelled, never guessed at.

    "unclassified" has to stay visible in the summary table: a file forced into
    a role would be counted as material it is not.
    """
    assert classify_env_file(name) == "unclassified"
