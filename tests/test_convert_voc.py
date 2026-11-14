import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_voc import convert

XML = """<annotation><filename>{name}</filename><size><width>100</width><height>50</height><depth>3</depth></size>
<object><name>hat</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>
<object><name>person</name><bndbox><xmin>90</xmin><ymin>20</ymin><xmax>120</xmax><ymax>40</ymax></bndbox></object>
<object><name>dog</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
<object><name>hat</name><bndbox><xmin>40</xmin><ymin>40</ymin><xmax>40</xmax><ymax>45</ymax></bndbox></object>
</annotation>"""


class ConvertVocTests(unittest.TestCase):
    def test_maps_classes_clips_boxes_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            voc = Path(folder) / "VOC"
            (voc / "JPEGImages").mkdir(parents=True)
            (voc / "Annotations").mkdir()
            for i in range(2):
                (voc / "JPEGImages" / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
                (voc / "Annotations" / f"img{i}.xml").write_text(XML.format(name=f"img{i}.jpg"), encoding="utf-8")
            (voc / "Annotations" / "orphan.xml").write_text(XML.format(name="missing.jpg"), encoding="utf-8")
            report = convert(voc, Path(folder) / "out", {"hat": "helmet", "person": "head"})
            self.assertEqual(report["images"], 2)
            self.assertEqual(report["skipped_images"], 1)
            self.assertEqual(report["objects"], {"helmet": 2, "head": 2})
            self.assertEqual(report["unmapped"], {"dog": 2})
            self.assertEqual(report["dropped_boxes"], 2)
            classes = (Path(folder) / "out" / "classes.txt").read_text(encoding="utf-8").split()
            self.assertEqual(classes, ["helmet", "head"])
            lines = (Path(folder) / "out" / "labels" / "img0.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "0 0.200000 0.400000 0.200000 0.400000")
            self.assertEqual(lines[1], "1 0.950000 0.600000 0.100000 0.400000")     # clipped at the right edge
            self.assertTrue((Path(folder) / "out" / "images" / "img1.jpg").exists())
            self.assertTrue(json.loads((Path(folder) / "out" / "conversion_report.json").read_text())["images"] == 2)
            limited = convert(voc, Path(folder) / "out2", {"hat": "helmet"}, limit=1)
            self.assertEqual(limited["images"], 1)
            with self.assertRaises(ValueError):
                convert(Path(folder) / "nowhere", Path(folder) / "out3", {"hat": "helmet"})


if __name__ == "__main__":
    unittest.main()
