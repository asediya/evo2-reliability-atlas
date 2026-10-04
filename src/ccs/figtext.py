# -*- coding: utf-8 -*-
"""Keep haloed annotations in the PDF text layer.

An annotation drawn over a plotted line needs a white halo to stay legible, and the halo is a
matplotlib path effect. Text carrying a path effect is rendered as vector outlines, so the string
leaves the PDF text layer entirely: it cannot be copied, searched or read aloud, and it is
recoverable only by rasterising the page.

Removing the halo would fix the text layer and cost legibility. Drawing a fully transparent twin of
the same string, at the same place, in the same font, costs nothing: the rendered page is unchanged
to the pixel and the text layer is complete again.

Call `restore_text_layer(fig)` once, immediately before `savefig`.

    from .figtext import restore_text_layer
    restore_text_layer(fig)
    fig.savefig(...)

Verified against the built figures: the strings come back out of `page.get_text()`, and a 600-dpi
raster of the page before and after is byte-identical.
"""
from matplotlib.text import Annotation


def _twin_props(t):
    return dict(
        color=t.get_color(),
        fontproperties=t.get_fontproperties(),
        horizontalalignment=t.get_horizontalalignment(),
        verticalalignment=t.get_verticalalignment(),
        rotation=t.get_rotation(),
        linespacing=getattr(t, "_linespacing", 1.2),
        zorder=t.get_zorder(),
        alpha=0.0,
    )


def _texts(fig):
    for ax in fig.get_axes():
        yield ax, list(ax.texts)
    yield fig, list(fig.texts)


def restore_text_layer(fig):
    """Add an invisible, extractable copy of every haloed string in `fig`.

    Returns the number of twins added, so a builder can assert on it if it wants to.
    """
    fig.canvas.draw()                       # an Annotation has no placement until it is laid out
    rend = fig.canvas.get_renderer()
    n = 0
    for owner, items in list(_texts(fig)):
        for t in items:
            if not t.get_path_effects() or (t.get_alpha() == 0.0) or not t.get_text():
                continue
            props = _twin_props(t)
            if isinstance(t, Annotation) or t.get_rotation():
                # Annotations place themselves through xycoords/textcoords rather than through the
                # artist transform, and a rotated string's own transform does not reproduce its
                # rendered box. Both are exact in display space, which is where they have already
                # been laid out. A rotated twin is placed unrotated inside its own rendered box:
                # invisible either way, and reading order is what the text layer is for.
                bb = t.get_window_extent(rend)
                # Back to figure fractions, not left in display pixels: the layout renderer runs at
                # the figure's own dpi and the PDF backend renders at 72, so a twin pinned in
                # display coordinates lands at a different place in the file than on the screen --
                # far enough, at dpi 100 against 72, to leave the page and vanish from the output.
                x, y = fig.transFigure.inverted().transform((bb.x0, bb.y0))
                props["rotation"] = 0
                props["horizontalalignment"] = "left"
                props["verticalalignment"] = "bottom"
                tw = fig.text(x, y, t.get_text(), **props)
            else:
                tw = fig.text(0, 0, t.get_text(), **props)
                tw.set_transform(t.get_transform())
                tw.set_position(t.get_position())
            n += 1
    return n
