#!/usr/bin/env python3
"""Expand the source xacro, never the potentially stale ROS install copy.

Run with ROS Humble's system Python. Only the generated offline URDF is changed:
its tool collisions deliberately use the current visual CAD placement, pending
physical validation. No live robot description is edited.
"""
import copy
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import xacro

HERE = Path(__file__).resolve().parent
WS = HERE.parents[4]
DESCRIPTION = WS / "src/ros2_kortex/kortex_description"


def main():
    with tempfile.TemporaryDirectory(prefix="curobo-source-") as directory:
        prefix = Path(directory)
        index = prefix / "share/ament_index/resource_index/packages"
        index.mkdir(parents=True)
        (index / "kortex_description").touch()
        (prefix / "share/kortex_description").symlink_to(DESCRIPTION)
        os.environ["AMENT_PREFIX_PATH"] = str(prefix) + os.pathsep + os.environ.get("AMENT_PREFIX_PATH", "")
        doc = xacro.process_file(str(DESCRIPTION / "robots/gen3.xacro"), mappings={
            "dof": "7", "gripper": "thesis_ee", "vision": "true",
            "use_fake_hardware": "true"})
        root = ET.fromstring(doc.toxml())
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            for source in (prefix / "share/kortex_description", DESCRIPTION):
                filename = filename.replace("file://" + str(source), "package://kortex_description")
            mesh.set("filename", filename)

    tool = root.find("./link[@name='thesis_ee']")
    for collision in list(tool.findall("collision")):
        tool.remove(collision)
    for visual in tool.findall("visual"):
        collision = ET.SubElement(tool, "collision")
        for tag in ("origin", "geometry"):
            collision.append(copy.deepcopy(visual.find(tag)))
    tip = root.find("./joint[@name='assembly_tip_joint']")
    assert tip.find("parent").get("link") == "bracelet_link"
    assert tuple(map(float, tip.find("origin").get("xyz").split())) == (0.108, -0.008, -0.411), "Tip calibration changed; review this offline model"
    ET.indent(root)
    output = HERE / "gen3_surgical.urdf"
    output.write_text('<!-- Generated from source xacro; OFFLINE tool collisions use visual CAD placement. -->\n' + ET.tostring(root, encoding="unicode") + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
