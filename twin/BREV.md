# Running the twin on Brev

Owners: Cade, Vlad. Brev bills while an instance runs, so stop it when you're done and add a line to `runs.csv`.

## Set up an instance

1. Deploy the [Isaac Launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46) from the website with the default compute (1× L40S, about $3.65/hr). After Brev says it's running, its containers take about 10 more minutes to build.
2. Install the Brev CLI (the instance page shows the command), then run `brev login` and `brev refresh`. `ssh <instance>` now works.
3. Let the container's user write to Isaac Sim's folder. Without this, renders come back empty and Isaac Sim crashes. It takes about 15 minutes on a fresh instance and prints nothing until it's done:
   ```bash
   ssh <instance> 'docker exec -u root vscode chown -R ubuntu:ubuntu /isaac-sim /workspace'
   ```
4. Copy the code, the fill station and the list of vials to render:
   ```bash
   uv run python twin/manifest.py && uv run python twin/fake_render.py
   INTO='docker exec -i vscode bash -c "mkdir -p /workspace/rockwell && tar -xzf - -C /workspace/rockwell"'
   git ls-files -co --exclude-standard | COPYFILE_DISABLE=1 tar --no-xattrs -czf - -T - | ssh <instance> "$INTO"
   COPYFILE_DISABLE=1 tar --no-xattrs -czhf - data/csi_digital_twin_v1/stations/fill_station synthetic/manifest.csv \
       synthetic/tank_still.jpg | ssh <instance> "$INTO"
   ```

## Render

Open the instance page's Shareable URL. It's VS Code in the browser; open a terminal there.

```bash
cd /workspace/rockwell
export ACCEPT_EULA=Y
/isaac-sim/python.sh twin/render.py --limit 12                              # 12 vials, a quick look
nohup /isaac-sim/python.sh twin/render.py > synthetic/render.log 2>&1 &    # all 462 vials, about 80 min
```

- The first run takes about 4 extra minutes to compile shaders.
- The `OmniHub` and `GLFW` warnings are harmless.
- `twin/material_lut.json` (committed) maps liquid colors to the camera. Rerun `/isaac-sim/python.sh twin/render.py --calibrate` only if you change the twin's material or lights.

## Bring results back and stop

```bash
ssh <instance> 'docker exec vscode tar -czf - -C /workspace/rockwell synthetic' | tar -xzf -
brev stop <instance>
```

`synthetic/` then loads with `dataset.frames("synthetic", 100)`.
