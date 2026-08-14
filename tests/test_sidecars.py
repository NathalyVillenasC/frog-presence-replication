"""Tests for reading metadata from XMP sidecars.

The sidecars are XML written by a cataloguing program, so each test writes a small file with
the same shape as the real ones. The cases covered are the ones that occur in the collection:
a sidecar naming one pond, one whose Location names both, one with no Location element at
all, and the free-text `Especie` fields, which are absent far more often than present.
"""

from pathlib import Path

import pytest

from frog_presence.sidecars import read_xmp_location, read_xmp_pond, read_xmp_species


def write_sidecar(tmp_path: Path, body: str) -> Path:
    """Write a minimal .xmp sidecar containing ``body`` and return its path."""
    path = tmp_path / "sample.xmp"
    path.write_text(
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        "<rdf:RDF xmlns:Iptc4xmpCore='http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/'>"
        f"{body}"
        "</rdf:RDF></x:xmpmeta>",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "location, pond",
    [
        ("Poza 1", "P1"),
        ("Poza 2", "P2"),
        ("Poza1", "P1"),
        ("Yasuni, poza 2, Ecuador", "P2"),
    ],
)

def test_a_stated_pond_is_read(tmp_path, location, pond):
    """Expected case: the pond is read whatever surrounds it in the field."""
    path = write_sidecar(
        tmp_path, f"<Iptc4xmpCore:Location>{location}</Iptc4xmpCore:Location>"
    )

    assert read_xmp_pond(path) == pond


def test_a_location_naming_both_ponds_returns_nothing(tmp_path):
    """Edge case: full cassette sides are catalogued as "Pozas", plural.

    A side master holds both ponds, so its sidecar names neither. Returning None here is
    the correct answer, not a failure to parse: guessing a pond for these files would
    invent a label for 82 recordings.
    """
    path = write_sidecar(
        tmp_path, "<Iptc4xmpCore:Location>Pozas, Yasuni</Iptc4xmpCore:Location>"
    )

    assert read_xmp_pond(path) is None


def test_a_sidecar_with_no_location_returns_nothing(tmp_path):
    """Failure case: the element is absent, so there is nothing to read.

    No default and no guess — the caller has to see that this file says nothing.
    """
    path = write_sidecar(tmp_path, "<Iptc4xmpCore:CountryCode>EC</Iptc4xmpCore:CountryCode>")

    assert read_xmp_pond(path) is None


def test_an_empty_location_returns_nothing(tmp_path):
    """Failure case: the element exists but is empty."""
    path = write_sidecar(tmp_path, "<Iptc4xmpCore:Location></Iptc4xmpCore:Location>")

    assert read_xmp_pond(path) is None


def test_the_raw_location_is_returned_untouched(tmp_path):
    """Expected case: the full string is available, not only the pond read out of it.

    Explaining why 95 sidecars state no pond needs the text itself, so the reader and the
    interpreter are kept apart.
    """
    path = write_sidecar(
        tmp_path,
        "<Iptc4xmpCore:Location>  Estación Científica Yasuní (PUCE), Poza 1  "
        "</Iptc4xmpCore:Location>",
    )

    assert read_xmp_location(path) == "Estación Científica Yasuní (PUCE), Poza 1"
    assert read_xmp_pond(path) == "P1"


def test_a_location_with_no_pond_is_still_returned(tmp_path):
    """Edge case: the two readers disagree, and both answers are correct.

    A cassette side is catalogued as "Pozas": there is text to show, but no pond to read.
    """
    path = write_sidecar(tmp_path, "<Iptc4xmpCore:Location>Pozas</Iptc4xmpCore:Location>")

    assert read_xmp_location(path) == "Pozas"
    assert read_xmp_pond(path) is None


def test_a_missing_location_reads_as_nothing(tmp_path):
    """Failure case: no element, so neither reader has anything to return."""
    path = write_sidecar(tmp_path, "<photoshop:Country>Ecuador</photoshop:Country>")

    assert read_xmp_location(path) is None
    assert read_xmp_pond(path) is None


def user_fields(*entries: str) -> str:
    """Wrap free-text entries, such as species, collector, or date values entered during cataloguing, 

    in the same `UserFields` XML structure used by the cataloguing program."""
    
    items = "".join(f"<rdf:li>{entry}</rdf:li>" for entry in entries)
    return f"<expressionmedia:UserFields><rdf:Bag>{items}</rdf:Bag></expressionmedia:UserFields>"


def test_the_species_named_in_a_sidecar_are_read(tmp_path):
    """Expected case: every Especie entry is returned, lowercased and sorted."""
    path = write_sidecar(
        tmp_path,
        user_fields("Colector=SRR", "Especie=Hypsiboas cinerascens", "Especie 2=Boana sp"),
    )

    assert read_xmp_species(path) == ["boana sp", "hypsiboas cinerascens"]


def test_the_same_species_written_twice_is_returned_once(tmp_path):
    """Edge case: a name repeated across fields is one species, not two.

    The result feeds a count of how many files name each species, so a duplicate inside a
    single sidecar would inflate that count.
    """
    path = write_sidecar(
        tmp_path, user_fields("Especie=Boana sp", "Especie 2=boana sp")
    )

    assert read_xmp_species(path) == ["boana sp"]


def test_an_empty_especie_field_is_not_a_species(tmp_path):
    """Edge case: the field exists but was left blank by the cataloguer."""
    path = write_sidecar(tmp_path, user_fields("Especie=", "Especie 2=   "))

    assert read_xmp_species(path) == []


def test_a_sidecar_with_no_especie_field_names_nothing(tmp_path):
    """Failure case: most sidecars carry no species at all.

    The empty list is the honest answer, and it is what keeps the species evidence
    described as partial rather than complete.
    """
    path = write_sidecar(tmp_path, user_fields("Colector=SRR", "Fecha=2001-11-14"))

    assert read_xmp_species(path) == []
