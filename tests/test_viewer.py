import zipfile
from pathlib import Path

import pytest

from app import viewer


VOC_XML = """<annotation><filename>sample.jpg</filename><size><width>100</width><height>50</height></size><object><name> line_fault </name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>80</xmax><ymax>40</ymax></bndbox></object></annotation>"""


def make_archive(path: Path, xml: str = VOC_XML, image_path: str = "set/JPEGImages/sample.jpg") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("set/Annotations/sample.xml", xml)
        archive.writestr(image_path, b"not-used-by-parser")


def test_voc_record_preserves_source_label_and_converts_box(tmp_path):
    archive_path = tmp_path / "voc.zip"
    make_archive(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        record = viewer._xml_record(archive, archive.getinfo("set/Annotations/sample.xml"), set(archive.namelist()))
    assert record is not None
    image_path, payload = record
    assert image_path == "set/JPEGImages/sample.jpg"
    assert payload["annotations"] == [{"label": "line_fault", "bbox": [10.0, 5.0, 70.0, 35.0]}]


def test_voc_record_skips_missing_image(tmp_path):
    archive_path = tmp_path / "voc.zip"
    make_archive(archive_path, image_path="other.jpg")
    with zipfile.ZipFile(archive_path) as archive:
        assert viewer._xml_record(archive, archive.getinfo("set/Annotations/sample.xml"), set(archive.namelist())) is None


def test_voc_record_skips_empty_or_zero_dimension_annotations(tmp_path):
    archive_path = tmp_path / "voc.zip"
    no_size = """<annotation><filename>sample.jpg</filename><size><width>0</width><height>0</height></size><object><name>fault</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object></annotation>"""
    make_archive(archive_path, xml=no_size)
    with zipfile.ZipFile(archive_path) as archive:
        assert viewer._xml_record(archive, archive.getinfo("set/Annotations/sample.xml"), set(archive.namelist())) is None

    no_objects = """<annotation><filename>sample.jpg</filename><size><width>100</width><height>50</height></size></annotation>"""
    make_archive(archive_path, xml=no_objects)
    with zipfile.ZipFile(archive_path) as archive:
        assert viewer._xml_record(archive, archive.getinfo("set/Annotations/sample.xml"), set(archive.namelist())) is None


def test_voc_record_rejects_malformed_xml(tmp_path):
    archive_path = tmp_path / "voc.zip"
    make_archive(archive_path, xml="<annotation>")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Invalid VOC annotation"):
            viewer._xml_record(archive, archive.getinfo("set/Annotations/sample.xml"), set(archive.namelist()))


def test_voc_rejects_unsafe_zip_path(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.xml", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe path"):
            viewer._safe_members(archive)
