#!/usr/bin/env python3
from __future__ import annotations

import collections
import glob
import os
import re
import sys


def patch_chamfer_lazy() -> None:
    targets = [
        "models/ldif/testing.py",
        "models/multi_view/modules/stitching_stage.py",
        "models/mgnet/loss.py",
    ]
    for path in targets:
        if not os.path.exists(path):
            continue
        s = open(path).read()
        if "DPC_PATCH_CHAMFER" in s:
            print(f"[chamfer] already patched: {path}")
            continue
        s = re.sub(
            r"^from external\.pyTorchChamferDistance\.chamfer_distance import ChamferDistance.*$",
            "# DPC_PATCH_CHAMFER\ntry:\n"
            "    from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance\n"
            "except Exception as _e:\n"
            '    print("[chamfer disabled]", _e)\n    ChamferDistance = None',
            s,
            flags=re.M,
        )
        s = s.replace(
            "dist_chamfer = ChamferDistance()",
            "dist_chamfer = ChamferDistance() if ChamferDistance is not None else None",
        )
        open(path, "w").write(s)
        print(f"[chamfer] patched: {path}")


def patch_checkpoint_keys() -> None:
    found = glob.glob("out/**/model_best.pth", recursive=True)
    if not found:
        print("[ckpt] model_best.pth not found (weights not extracted yet?)")
        return
    try:
        import torch
    except ImportError:
        print("[ckpt] torch unavailable — run the script inside the Pano3D env "
              "(conda run -n Pano3D python ...), otherwise checkpoint keys stay unfixed")
        return
    for ckpt_path in found:
        ck = torch.load(ckpt_path, map_location="cpu")
        if not (isinstance(ck, dict) and "net" in ck):
            continue
        net = ck["net"]
        keys = list(net.keys())
        if keys and ".module." in keys[0]:
            new = collections.OrderedDict()
            for k, v in net.items():
                new[k.replace(".module.", ".")] = v
            ck["net"] = new
            torch.save(ck, ckpt_path)
            print(f"[ckpt] {ckpt_path}: renamed {len(new)} keys")
        else:
            print(f"[ckpt] {ckpt_path}: keys already without prefix")


def patch_empty_scene() -> None:
    p = "models/pano3d/modules/method.py"
    if not os.path.exists(p):
        return
    s = open(p).read()
    if "DPC_PATCH_EMPTY_SCENE" in s:
        print(f"[empty_scene] already patched: {p}")
        return
    old = (
        "        est_objs = est_data.get('objs')\n"
        "        if est_objs:\n"
        "            if hasattr(self, 'shape_encoder'):"
    )
    new = (
        "        est_objs = est_data.get('objs')\n"
        "        # DPC_PATCH_EMPTY_SCENE: skip scenes with no detected objects\n"
        "        if est_objs and 'rgb' in est_objs:\n"
        "            if hasattr(self, 'shape_encoder'):"
    )
    if old in s:
        open(p, "w").write(s.replace(old, new))
        print(f"[empty_scene] patched: {p}")
    else:
        print(f"[empty_scene] anchor not found in {p} (different version?) — skip")


def patch_safe_viz() -> None:
    p = "models/pano3d/testing.py"
    if not os.path.exists(p):
        return
    src = open(p).read()
    if "DPC_PATCH_SAFE_VIZ" in src:
        print(f"[safe_viz] already patched: {p}")
        return
    start = src.find("    def visualize_step(self, est_data):")
    if start < 0:
        print(f"[safe_viz] visualize_step not found in {p} — skip")
        return
    rest = src[start + 10:]
    end_match = re.search(r"\n    def \w+", rest)
    end = start + 10 + (end_match.start() if end_match else len(rest))
    new_func = (
        "    def visualize_step(self, est_data):\n"
        "        # DPC_PATCH_SAFE_VIZ: save data.pkl before attempting the 3D render\n"
        "        if 'objs' in est_data and 'mesh' not in est_data['objs'] \\\n"
        "                and 'mesh_extractor' in est_data['objs'] and self.cfg.config['full']:\n"
        "            est_data['objs']['mesh'] = est_data['objs']['mesh_extractor'].extract_mesh()\n"
        "        est_scenes = IGScene.from_batch(est_data)\n\n"
        "        for est_scene in est_scenes:\n"
        "            scene_folder = os.path.join(self.cfg.config['log']['vis_path'], est_scene['scene'], est_scene['name'])\n"
        "            os.makedirs(scene_folder, exist_ok=True)\n"
        "            try:\n"
        "                est_scene.to_pickle(scene_folder)\n"
        "                print(f'[DPC_PATCH] data.pkl saved -> {scene_folder}')\n"
        "            except Exception as _e:\n"
        "                print(f'[DPC_PATCH] to_pickle failed: {_e}')\n"
        "            # DPC_PATCH_EXPORT_OBJ_MESH: per-object CANONICAL meshes -> obj_mesh/<i>.ply (shape metric)\n"
        "            try:\n"
        "                if self.cfg.config.get('full') and est_scene.mesh_io is not None and len(est_scene.mesh_io):\n"
        "                    _omd = os.path.join(scene_folder, 'obj_mesh')\n"
        "                    os.makedirs(_omd, exist_ok=True)\n"
        "                    for _i, _m in est_scene.mesh_io.items():\n"
        "                        _m.export(os.path.join(_omd, '%s.ply' % _i))\n"
        "                    print('[DPC_PATCH] obj_mesh x%d -> %s' % (len(est_scene.mesh_io), _omd))\n"
        "            except Exception as _e:\n"
        "                print('[DPC_PATCH] obj_mesh export failed: %s' % _e)\n"
        "            try:\n"
        "                gpu_id = int(self.cfg.config['device']['gpu_ids'].split(',')[0])\n"
        "                visualizer = IGVisualizer(est_scene, gpu_id=gpu_id)\n"
        "                if self.cfg.config['log'].get('save_mesh'):\n"
        "                    scene_mesh = est_scene.merge_mesh(\n"
        "                        colorbox=visualizer.colorbox / 255,\n"
        "                        separate=self.cfg.config['log'].get('separate_mesh', True)\n"
        "                    )\n"
        "                    if scene_mesh is not None:\n"
        "                        scene_mesh.export(os.path.join(scene_folder, 'mesh.glb'))\n"
        "                from PIL import Image as PILImage\n"
        "                image = visualizer.image(est_scene['image_np'])\n"
        "                image_render = visualizer.image(image=visualizer.render(background=200))\n"
        "                PILImage.fromarray(image).save(os.path.join(scene_folder, 'rgb.png'))\n"
        "                print(f'[DPC_PATCH] visualization OK -> {scene_folder}')\n"
        "            except Exception as _e:\n"
        "                print(f'[DPC_PATCH_SAFE_VIZ] viz skipped: {type(_e).__name__}: {str(_e)[:120]}')\n\n"
    )
    open(p, "w").write(src[:start] + new_func + src[end:])
    print(f"[safe_viz] patched: {p}")


def patch_collate_contour_clip() -> None:
    p = "models/pano3d/dataloader.py"
    if not os.path.exists(p):
        return
    s = open(p).read()
    if "'contour': {'x': True, 'y': True}, 'contour_clip'" in s:
        print(f"[collate] contour_clip already in force_list: {p}")
        return
    old = "{'contour': {'x': True, 'y': True}}"
    n = s.count(old)
    if n == 0:
        print(f"[collate] contour anchor not found in {p} — skip")
        return
    new = "{'contour': {'x': True, 'y': True}, 'contour_clip': {'x': True, 'y': True}}"
    open(p, "w").write(s.replace(old, new))
    print(f"[collate] contour_clip added to force_list ({n})")


def patch_eval_metrics_degenerate() -> None:
    p = "models/eval_metrics.py"
    if not os.path.exists(p):
        return
    s = open(p).read()
    changed = False

    if "DPC_PATCH: degenerate 3D" not in s:
        g_old = "    # 2D projection on the horizontal plane (x-y plane)"
        if g_old in s:
            s = s.replace(
                g_old,
                "    # DPC_PATCH: degenerate 3D bbox -> IoU 0 (miss), don't crash eval\n"
                "    import numpy as _np\n"
                "    if not (_np.all(_np.isfinite(cu1)) and _np.all(_np.isfinite(cu2))):\n"
                "        return 0.0\n" + g_old,
                1,
            )
            changed = True
        r_old = "    return inter_vol / (vol1 + vol2 - inter_vol)"
        if r_old in s:
            s = s.replace(
                r_old,
                "    _union = vol1 + vol2 - inter_vol\n"
                "    return inter_vol / _union if _union > 0 else 0.0",
                1,
            )
            changed = True

    if "DPC_PATCH: degenerate 2D" not in s:
        pat = re.compile(
            r"[ \t]*assert bb1\['u1'\] <= bb1\['u2'\]\n"
            r"[ \t]*assert bb1\['v1'\] <= bb1\['v2'\]\n"
            r"[ \t]*assert bb2\['u1'\] <= bb2\['u2'\]\n"
            r"[ \t]*assert bb2\['v1'\] <= bb2\['v2'\]\n"
        )
        guard = (
            "    # DPC_PATCH: degenerate 2D bbox -> IoU 0 (miss), don't assert-crash eval\n"
            "    if not (bb1['u1'] <= bb1['u2'] and bb1['v1'] <= bb1['v2']\n"
            "            and bb2['u1'] <= bb2['u2'] and bb2['v1'] <= bb2['v2']):\n"
            "        return 0.0\n"
        )
        s, n = pat.subn(guard, s, count=1)
        if n:
            changed = True

    if changed:
        open(p, "w").write(s)
        print(f"[eval] degenerate-box guard added: {p}")
    else:
        print(f"[eval] already patched / anchors not found: {p}")


def patch_num_workers() -> None:
    configs = [
        "configs/first_stage_igibson.yaml",
        "configs/relation_scene_gcn_igibson.yaml",
        "configs/pano3d_igibson.yaml",
    ]
    for p in configs:
        if not os.path.exists(p):
            continue
        s = open(p).read()
        s2 = re.sub(r"(\n\s*num_workers:\s*)\d+", r"\g<1>0", s)
        if s2 != s:
            open(p, "w").write(s2)
            print(f"[num_workers] -> 0 in {p}")


def patch_eval_no_grad() -> None:
    p = "models/ldif/training.py"
    if not os.path.exists(p):
        return
    s = open(p).read()
    if "DPC_PATCH_EVAL_NOGRAD" in s:
        print(f"[eval_nograd] already patched: {p}")
        return
    if "import torch" not in s.split("class Trainer")[0]:
        s = s.replace(
            "from models.training import BaseTrainer",
            "import torch\nfrom models.training import BaseTrainer",
            1,
        )
    old = (
        "        loss = self.compute_loss(data)\n"
        "        loss['total'] = loss['total'].item()\n"
        "        return loss"
    )
    if old not in s:
        print(f"[eval_nograd] eval_step anchor not found in {p} — skip")
        return
    new = (
        "        # DPC_PATCH_EVAL_NOGRAD: validation without the autograd graph (else OOM on MGN eval)\n"
        "        with torch.no_grad():\n"
        "            loss = self.compute_loss(data)\n"
        "        loss['total'] = loss['total'].item()\n"
        "        return loss"
    )
    open(p, "w").write(s.replace(old, new, 1))
    print(f"[eval_nograd] patched: {p}")


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    repo = os.path.abspath(repo)
    if not os.path.exists(os.path.join(repo, "main.py")):
        print(f"ERROR: {repo} does not look like a DPC repo (no main.py).")
        print("Run from the cloned DPC root or pass the path as an argument.")
        return 2
    os.chdir(repo)
    print(f"Applying DPC patches in: {repo}\n")
    patch_chamfer_lazy()
    patch_empty_scene()
    patch_safe_viz()
    patch_collate_contour_clip()
    patch_eval_metrics_degenerate()
    patch_num_workers()
    patch_eval_no_grad()
    print("\nDone. DPC inference/evaluation can be run now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
