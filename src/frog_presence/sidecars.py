"""Reading the Adobe XMP sidecars that travel next to the pond recordings.

Each recording may be accompanied by an ``.xmp`` file holding the metadata its cataloguer
wrote: where it was recorded, when, by whom, and sometimes which species were heard. Two of
those fields matter to the replication, and for both of them the sidecar is the only place
they are written down.
"""

import re
from pathlib import Path


LOCATION = re.compile(
    r"<Iptc4xmpCore:Location>(.*?)</Iptc4xmpCore:Location>", re.S
)


def read_xmp_location(path: Path) -> str | None:
    """Return the raw Location string of a sidecar, whatever it happens to say.

    Args:
        path (Path): An .xmp sidecar file.

    Returns:
        str | None: The Location text, or None when the element is absent.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    match = LOCATION.search(text)
    return match.group(1).strip() if match else None


def read_xmp_pond(path: Path) -> str | None:
    """Read the pond a recording belongs to from its .xmp sidecar.

    Reads the same field as ``read_xmp_location`` and interprets it: most Location strings
    name a pond, but the ones belonging to full cassette sides name both.

    Args:
        path (Path): An .xmp sidecar file.

    Returns:
        str | None: "P1" or "P2" when the Location field states one, otherwise None.
    """
    location = read_xmp_location(path)

    if not location:
        return None

    pond = re.search(r"poza\s*([12])", location, re.I)

    return f"P{pond.group(1)}" if pond else None


def read_xmp_species(path: Path) -> list:
    """Return the species named in a sidecar's `Especie` fields.

    Args:
        path (Path): An .xmp sidecar file.

    Returns:
        list[str]: Distinct species names, lowercased. Empty when the field is absent.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return sorted({match.group(1).strip().lower()
                   for match in re.finditer(r"<rdf:li>Especie[^=]*=([^<]*)</rdf:li>", text)
                   if match.group(1).strip()})
