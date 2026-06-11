"""
Composable render graph for ChimeraX scene rendering.

The render graph replaces the monolithic _draw_scene method with an ordered
list of RenderPass objects.  Each pass declares when it should run (enabled)
and what it does (execute).  The graph splits passes into two phases:

  pre_view  — run once per frame (shadow map generation)
  per_view  — run once per camera view (stereo left/right, cubemap faces)

Adding a new post-processing effect (SSAO, bloom, …) is just inserting a
new pass object at the right position.
"""


class ResourcePool:
    """Named GPU resource cache owned by the RenderGraph.

    Passes create resources using backend-specific code, then store them
    here by name.  The pool tracks dimensions so passes can detect when a
    resize is needed without inspecting the GPU object itself.
    """

    __slots__ = ('_entries',)

    def __init__(self):
        self._entries = {}   # name → (resource, width, height)

    def get(self, name):
        """Return the resource stored under *name*, or None."""
        entry = self._entries.get(name)
        return entry[0] if entry is not None else None

    def put(self, name, resource, *, width=0, height=0):
        """Store *resource* under *name* with optional dimensions."""
        self._entries[name] = (resource, width, height)

    def needs_resize(self, name, width, height):
        """True if *name* is absent or has different dimensions."""
        entry = self._entries.get(name)
        if entry is None:
            return True
        return entry[1] != width or entry[2] != height

    def remove(self, name):
        """Remove and return the resource (caller deletes GPU object)."""
        entry = self._entries.pop(name, None)
        return entry[0] if entry is not None else None

    def clear(self):
        """Remove all entries.  Does NOT delete GPU objects."""
        self._entries.clear()


class PassContext:
    """Mutable state bag passed to every render pass."""

    __slots__ = (
        'renderer', 'camera', 'view',
        'camera_position', 'view_num',
        'opaque_drawings', 'transparent_drawings',
        'highlight_drawings', 'on_top_drawings',
        'shadow_enabled', 'multishadow_enabled',
        # Subsystem state detected once, used by multiple passes.
        'offscreen', 'silhouette',
        # Persistent resource pool (lives on the RenderGraph).
        'resources',
    )

    def __init__(self, *, renderer, camera, view,
                 opaque_drawings, transparent_drawings,
                 highlight_drawings, on_top_drawings):
        self.renderer = renderer
        self.camera = camera
        self.view = view
        self.camera_position = None   # Set per-view by RenderGraph
        self.view_num = 0

        self.opaque_drawings = opaque_drawings
        self.transparent_drawings = transparent_drawings
        self.highlight_drawings = highlight_drawings
        self.on_top_drawings = on_top_drawings

        self.shadow_enabled = False
        self.multishadow_enabled = False

        # Resolved once per frame in RenderGraph.execute().
        self.offscreen = None
        self.silhouette = None

        # Set by RenderGraph.execute() — reference to graph.resources.
        self.resources = None

    @property
    def no_drawings(self):
        return (len(self.opaque_drawings) == 0
                and len(self.transparent_drawings) == 0
                and len(self.highlight_drawings) == 0
                and len(self.on_top_drawings) == 0)


class RenderPass:
    """Base class for a single composable render pass."""

    name = None  # Subclasses must set this.

    def enabled(self, ctx):
        """Return whether this pass should execute."""
        return True

    def execute(self, ctx):
        """Perform the rendering work."""
        raise NotImplementedError


class RenderGraph:
    """Ordered list of render passes executed each frame."""

    def __init__(self):
        self._pre_view_passes = []
        self._per_view_passes = []
        self.resources = ResourcePool()

    # -- Construction --------------------------------------------------------

    def add_pre_view(self, p):
        self._pre_view_passes.append(p)

    def add_per_view(self, p):
        self._per_view_passes.append(p)

    def remove(self, name):
        self._pre_view_passes = [p for p in self._pre_view_passes
                                 if p.name != name]
        self._per_view_passes = [p for p in self._per_view_passes
                                 if p.name != name]

    def insert_after(self, after_name, p):
        for lst in (self._pre_view_passes, self._per_view_passes):
            for i, existing in enumerate(lst):
                if existing.name == after_name:
                    lst.insert(i + 1, p)
                    return
        raise KeyError(f'No pass named {after_name!r}')

    def insert_before(self, before_name, p):
        for lst in (self._pre_view_passes, self._per_view_passes):
            for i, existing in enumerate(lst):
                if existing.name == before_name:
                    lst.insert(i, p)
                    return
        raise KeyError(f'No pass named {before_name!r}')

    # -- Execution -----------------------------------------------------------

    def execute(self, ctx):
        """Run all passes for one frame."""
        ctx.resources = self.resources
        self._resolve_subsystems(ctx)

        # Pre-view passes (shadow maps).
        for p in self._pre_view_passes:
            if p.enabled(ctx):
                p.execute(ctx)

        # Per-view passes.
        camera = ctx.camera
        r = ctx.renderer
        for vnum in range(camera.number_of_views()):
            ctx.view_num = vnum
            camera.set_render_target(vnum, r)

            if ctx.no_drawings:
                camera.draw_background(vnum, r)
                continue

            for p in self._per_view_passes:
                if p.enabled(ctx):
                    p.execute(ctx)

    @staticmethod
    def _resolve_subsystems(ctx):
        """Detect which optional subsystems are active this frame."""
        r = ctx.renderer

        offscreen = r.offscreen if r.offscreen.enabled else None
        if ctx.highlight_drawings and r.outline.offscreen_outline_needed:
            offscreen = r.offscreen
        if offscreen and r.current_framebuffer() is not r.default_framebuffer():
            offscreen = None
        ctx.offscreen = offscreen

        if hasattr(r, '_texture_window'):
            # OpenGL: the view's Silhouette object both holds the
            # parameters and implements the drawing.
            silhouette = ctx.view.silhouette
        else:
            # Metal/Vulkan: the renderer carries its own implementation
            # (driven by the view's Silhouette parameter object), or has
            # none and silhouettes are unavailable.
            silhouette = getattr(r, '_silhouette_impl', None)
            if silhouette is not None:
                silhouette.params = ctx.view.silhouette
        ctx.silhouette = silhouette
