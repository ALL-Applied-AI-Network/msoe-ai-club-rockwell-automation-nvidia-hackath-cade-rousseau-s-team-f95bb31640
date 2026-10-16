"""Print a USD stage's prim tree, cameras, materials and animated attributes. No GPU needed.

    uv run --with usd-core python twin/inspect_usd.py data/csi_digital_twin_v1/stations/fill_station/fill_station.usda
"""
import sys

from pxr import Usd, UsdGeom, UsdShade

stage = Usd.Stage.Open(sys.argv[1])
print("time codes", stage.GetStartTimeCode(), "-", stage.GetEndTimeCode(), "up", UsdGeom.GetStageUpAxis(stage))

for prim in stage.Traverse():
    depth = str(prim.GetPath()).count("/") - 1
    line = "  " * depth + f"{prim.GetName()} [{prim.GetTypeName()}]"
    mat = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()
    if mat:
        line += f" mat={mat}"
    animated = [a.GetName() for a in prim.GetAttributes() if a.GetNumTimeSamples() > 1]
    if animated:
        line += f" animated={animated}"
    labels = {a.GetName()[15:]: a.Get() for a in prim.GetAttributes() if a.GetName().startswith("userProperties:")}
    if labels:
        line += f" {labels}"
    print(line)

print("\n== materials")
for prim in stage.Traverse():
    if prim.IsA(UsdShade.Shader):
        inputs = {i.GetBaseName(): i.Get() for i in UsdShade.Shader(prim).GetInputs() if i.Get() is not None}
        print(prim.GetPath(), UsdShade.Shader(prim).GetIdAttr().Get(), inputs)
