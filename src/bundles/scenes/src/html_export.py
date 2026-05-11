# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
# === UCSF ChimeraX Copyright ===

"""
Export scenes to standalone HTML pages that embed glTF geometry.

Two flavors:

- ``export_scene_html(session, scene_name, path)`` writes a single .html file
  with the scene's glb embedded inline as base64.

- ``export_storyboard_html(session, path, scene_names=None)`` writes a
  directory containing ``index.html`` (a small picker) plus one
  ``scene_NNN.html`` per scene. The picker embeds each scene on demand via
  an ``<object>`` tag.

Everything is designed to work straight from ``file://`` — there is no
``fetch()``, no .glb sidecar, and no local web server required. (three.js is
still loaded from a CDN over HTTPS, which browsers permit from file:// pages
and cache after first load.)
"""

import base64
import json
import os
import re

from chimerax.core.errors import UserError

from .scene import Scene


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def export_scene_html(session, scene_name, path):
    scene = session.scenes.get_scene(scene_name)
    if scene is None:
        raise UserError(f"Scene '{scene_name}' does not exist")

    if not path.lower().endswith(".html"):
        path = path + ".html"
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    with _preserved_session_state(session):
        scene.restore_scene()
        glb_bytes = _get_glb_bytes(session)
        view_cfg = _scene_view_config(session)

    html = _render_scene_viewer_html(scene_name, glb_bytes, view_cfg)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    session.logger.info(f"Wrote scene HTML to {path}")


def export_storyboard_html(session, path, scene_names=None):
    if scene_names is None:
        scenes = list(session.scenes.get_scenes())
    else:
        scenes = []
        for name in scene_names:
            s = session.scenes.get_scene(name)
            if s is None:
                raise UserError(f"Scene '{name}' does not exist")
            scenes.append(s)
    if not scenes:
        raise UserError("No scenes to export")

    if path.lower().endswith(".html"):
        raise UserError(
            "Storyboard export needs a directory path, not a .html file "
            "(per-scene files are written alongside index.html)."
        )
    os.makedirs(path, exist_ok=True)

    entries = []
    with _preserved_session_state(session):
        for i, scene in enumerate(scenes):
            scene.restore_scene()
            glb_bytes = _get_glb_bytes(session)
            view_cfg = _scene_view_config(session)

            scene_html = _render_scene_viewer_html(
                scene.get_name(), glb_bytes, view_cfg
            )
            scene_filename = f"scene_{i:03d}.html"
            with open(os.path.join(path, scene_filename), "w", encoding="utf-8") as f:
                f.write(scene_html)

            entries.append({
                "name": scene.get_name(),
                "thumbnail": _normalize_thumbnail(scene.get_thumbnail()),
                "file": scene_filename,
            })

    index_html = _render_storyboard_index_html(entries)
    index_path = os.path.join(path, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    session.logger.info(f"Wrote storyboard ({len(entries)} scenes) to {index_path}")


# ---------------------------------------------------------------------------
# Session state preservation
# ---------------------------------------------------------------------------

class _preserved_session_state:
    """Context manager that snapshots the session into a temp Scene on enter
    and restores it on exit, so per-scene exports don't disturb the user's
    current view."""

    _TEMP_NAME = "__chimerax_html_export_temp__"

    def __init__(self, session):
        self.session = session
        self._snapshot = None

    def __enter__(self):
        # Build a Scene directly without registering it with the manager, so
        # we don't fire SAVED triggers or pollute the user's scene list.
        self._snapshot = Scene(self.session, self._TEMP_NAME)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._snapshot is not None:
            self._snapshot.restore_scene()
        return False


# ---------------------------------------------------------------------------
# Geometry + view extraction
# ---------------------------------------------------------------------------

def _get_glb_bytes(session):
    """Run write_gltf without a filename to get the encoded glb bytes."""
    from chimerax.gltf.gltf import write_gltf
    # Disable centering — we want geometry to stay in original world coords
    # so the saved camera in view_cfg points at it without an offset.
    return write_gltf(
        session, filename=None,
        models=None,
        center=False,
        center_each_node=False,
        preserve_transparency=True,
    )


def _scene_view_config(session):
    """Capture camera + background of the live session as a JSON-friendly
    dict. Called after a scene has been restored so the live view reflects
    that scene's saved state."""
    view = session.view
    cam = view.camera

    pos = cam.position
    origin = [float(x) for x in pos.origin()]
    forward = [float(x) for x in cam.view_direction()]
    # Place.axes() returns rows (x, y, z) in scene coords; row 1 is the
    # camera's local up vector.
    up = [float(x) for x in pos.axes()[1]]

    # OrbitControls re-orients the camera to look from position toward
    # ``target``, so target MUST lie on the captured view ray or the export
    # gets re-aimed and the framing shifts. Project the bounds center onto
    # the view ray to keep the camera direction faithful while still putting
    # the orbit pivot near the geometric center.
    bounds = view.drawing_bounds()
    if bounds is not None:
        bc = bounds.center()
        dx, dy, dz = bc[0] - origin[0], bc[1] - origin[1], bc[2] - origin[2]
        d = dx * forward[0] + dy * forward[1] + dz * forward[2]
        if d <= 0:
            d = 1.0  # Bounds center behind camera; fall back to unit step.
        target = [origin[i] + d * forward[i] for i in range(3)]
    else:
        target = [origin[i] + forward[i] for i in range(3)]

    bg = view.background_color
    bg_rgb = [float(bg[0]), float(bg[1]), float(bg[2])]

    return {
        "position": origin,
        "target": target,
        "up": up,
        # ChimeraX field_of_view is the *horizontal* FOV in degrees; three.js
        # uses *vertical* FOV. The exported viewer converts at runtime based
        # on canvas aspect so framing follows resizes.
        "hfov_deg": float(getattr(cam, "field_of_view", 30.0)),
        "background": bg_rgb,
    }


def _normalize_thumbnail(thumbnail_b64):
    # Scene.take_thumbnail uses codecs.encode(..., 'base64') which inserts
    # newlines every 76 chars. Strip whitespace so we can drop it straight
    # into a data: URL.
    return re.sub(r"\s+", "", thumbnail_b64 or "")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

# three.js + GLTFLoader + OrbitControls via importmap from a CDN. Browsers
# allow file:// pages to load HTTPS modules; only file://-to-file:// fetch is
# blocked, which is why we embed the glb inline rather than as a sidecar.

_THREE_VERSION = "0.160.0"

_VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #eee;
    font-family: system-ui, sans-serif; overflow: hidden; }
  #viewer { position: absolute; inset: 0; }
  #viewer canvas { display: block; }
  #status { position: absolute; left: 12px; top: 8px; padding: 4px 8px;
    background: rgba(0,0,0,0.5); border-radius: 4px; font-size: 13px;
    pointer-events: none; opacity: 0; transition: opacity 0.2s; }
  #status.visible { opacity: 1; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@__THREE_VERSION__/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@__THREE_VERSION__/examples/jsm/"
  }
}
</script>
</head>
<body>
  <div id="viewer"><div id="status">Loading...</div></div>
  <script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const VIEW = __VIEW_JSON__;
const GLB_B64 = "__GLB_B64__";

const viewerEl = document.getElementById('viewer');
const statusEl = document.getElementById('status');

// logarithmicDepthBuffer copes with the wide far/near ratio we get when
// the camera dollies far out from a small scene, where a linear depth
// buffer would Z-fight badly across distant geometry.
const renderer = new THREE.WebGLRenderer({
    antialias: true,
    logarithmicDepthBuffer: true,
});
renderer.setPixelRatio(window.devicePixelRatio);
viewerEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// Procedural room environment provides soft, all-direction ambient that
// makes PBR materials read with smooth gradients instead of the flat,
// posterized look of an AmbientLight. Generated once at startup from
// three.js's built-in synthetic room (no external HDR asset required).
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
pmrem.dispose();

// Headlight that follows the camera — matches ChimeraX's default lighting
// where the key light is locked to the camera frame. Position is updated
// each tick so the highlight tracks the user's view as they orbit.
const headlight = new THREE.DirectionalLight(0xffffff, 1.2);
scene.add(headlight);
scene.add(headlight.target);

// Small ambient floor so deeply self-shadowed parts of the scene still
// register (the env map already provides most ambient, but this lifts
// the very-darkest pixels a touch).
scene.add(new THREE.AmbientLight(0xffffff, 0.2));

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function applyView() {
    const w = viewerEl.clientWidth, h = Math.max(1, viewerEl.clientHeight);
    const aspect = w / h;
    // Convert horizontal FOV (ChimeraX) to vertical FOV (three.js) for the
    // current aspect so framing matches what the user saw in ChimeraX.
    const hfov = (VIEW.hfov_deg || 30) * Math.PI / 180;
    const vfov = 2 * Math.atan(Math.tan(hfov / 2) / aspect);
    camera.fov = vfov * 180 / Math.PI;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    camera.position.set(...VIEW.position);
    camera.up.set(...VIEW.up);
    controls.target.set(...VIEW.target);
    controls.update();
    const bg = VIEW.background;
    renderer.setClearColor(new THREE.Color(bg[0], bg[1], bg[2]), 1.0);
}

function resize() {
    const w = viewerEl.clientWidth, h = viewerEl.clientHeight;
    // Pass updateStyle=true (default) so three.js sets the canvas's inline
    // CSS width/height to match the framebuffer. Otherwise the canvas stays
    // at its default 300x150 CSS size and the rendered scene gets clipped
    // by the viewer container.
    renderer.setSize(w, h);
    applyView();
}
window.addEventListener('resize', resize);
resize();

function base64ToArrayBuffer(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
}

const loader = new GLTFLoader();
loader.parse(base64ToArrayBuffer(GLB_B64), '', (gltf) => {
    scene.add(gltf.scene);
    statusEl.classList.remove('visible');
    statusEl.textContent = '';
}, (err) => {
    console.error(err);
    statusEl.textContent = 'Failed to parse glTF';
    statusEl.classList.add('visible');
});
statusEl.classList.add('visible');

(function tick() {
    controls.update();
    // Keep the headlight pinned to the camera so highlights follow the
    // user's viewpoint as they orbit, matching ChimeraX's default rig.
    headlight.position.copy(camera.position);
    headlight.target.position.copy(controls.target);
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
})();
  </script>
</body>
</html>
"""


_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scene Storyboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #eee;
    font-family: system-ui, sans-serif; overflow: hidden; }
  #app { display: flex; flex-direction: column; height: 100%; }
  #strip { display: flex; flex-direction: row; gap: 6px; padding: 8px;
    background: #1a1a1a; border-bottom: 1px solid #2a2a2a;
    overflow-x: auto; overflow-y: hidden; flex: 0 0 auto; }
  .item { width: 110px; flex: 0 0 auto; padding: 4px;
    border: 2px solid transparent; border-radius: 4px; cursor: pointer;
    background: #222; }
  .item:hover { background: #2a2a2a; }
  .item.active { border-color: #4a90e2; background: #2c4a78; }
  .item img { display: block; width: 100%; height: auto; border-radius: 2px; }
  .item .name { font-size: 11px; margin-top: 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #main { flex: 1; position: relative; min-height: 0; }
  #main object, #main iframe { width: 100%; height: 100%; border: 0;
    display: block; }
  #help { position: absolute; right: 10px; bottom: 10px; font-size: 12px;
    color: #888; background: rgba(0,0,0,0.4); padding: 4px 8px;
    border-radius: 4px; pointer-events: none; }
</style>
</head>
<body>
  <div id="app">
    <div id="strip"></div>
    <div id="main"><div id="help">← → to navigate</div></div>
  </div>
  <script>
const SCENES = __SCENES_JSON__;
const stripEl = document.getElementById('strip');
const mainEl = document.getElementById('main');
let activeIdx = -1;

function show(i) {
    if (i < 0 || i >= SCENES.length || i === activeIdx) return;
    activeIdx = i;
    Array.from(stripEl.children).forEach((el, j) => {
        el.classList.toggle('active', j === i);
    });
    // Use <object> rather than <iframe> for parity with the legacy Chimera
    // exporter; both work from file:// without a server.
    mainEl.innerHTML = '';
    const obj = document.createElement('object');
    obj.type = 'text/html';
    obj.data = SCENES[i].file;
    mainEl.appendChild(obj);
    const help = document.createElement('div');
    help.id = 'help';
    help.textContent = '← → to navigate';
    mainEl.appendChild(help);
}

SCENES.forEach((s, i) => {
    const item = document.createElement('div');
    item.className = 'item';
    const img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + s.thumbnail;
    img.alt = s.name;
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = s.name;
    item.appendChild(img);
    item.appendChild(name);
    item.addEventListener('click', () => show(i));
    stripEl.appendChild(item);
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { show(Math.max(0, activeIdx - 1)); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { show(Math.min(SCENES.length - 1, activeIdx + 1)); e.preventDefault(); }
});

if (SCENES.length > 0) show(0);
  </script>
</body>
</html>
"""


def _render_scene_viewer_html(scene_name, glb_bytes, view_cfg):
    glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
    view_json = _safe_json(view_cfg)
    html = _VIEWER_TEMPLATE
    html = html.replace("__TITLE__", _html_escape(scene_name))
    html = html.replace("__THREE_VERSION__", _THREE_VERSION)
    html = html.replace("__VIEW_JSON__", view_json)
    html = html.replace("__GLB_B64__", glb_b64)
    return html


def _render_storyboard_index_html(entries):
    return _INDEX_TEMPLATE.replace("__SCENES_JSON__", _safe_json(entries))


def _safe_json(obj):
    """JSON-encode and escape any "</" sequences so the literal can never
    close the surrounding <script> tag (e.g. if a scene name contains
    "</script>")."""
    return json.dumps(obj).replace("</", "<\\/")


_HTML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}

def _html_escape(s):
    return re.sub(r"[&<>\"']", lambda m: _HTML_ESCAPES[m.group(0)], s)
