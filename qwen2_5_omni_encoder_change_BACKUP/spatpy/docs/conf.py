# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

import git
import dataclasses

DOC_DIR = os.path.dirname(__file__)

SPATPY_DIR = os.path.abspath(os.path.join(DOC_DIR, "..", "spatpy"))


EXCLUDE_PATTERNS = [os.path.join(SPATPY_DIR, s) for s in ("*test_*",)]

# IGNORE_FILES = {
#     "cicero": ["MakePassiveBeam"],
#     "fusion": ["design"],
# }

# for pkg, files in IGNORE_FILES.items():
#     for f in files:
#         EXCLUDE_PATTERNS.append(os.path.join(SPATPY_DIR, pkg, f + ".py"))

import spatpy

# -- Project information -----------------------------------------------------

project = "spatpy"
copyright = "2021 Audio Capture Research, Dolby"
author = "Audio Capture Research"

# The full version, including alpha/beta/rc tags
repo = git.Repo(search_parent_directories=True)

BRANCH_NAME = os.getenv("CI_COMMIT_REF_SLUG", None)
if BRANCH_NAME is None:
    BRANCH_NAME = repo.active_branch.name
    SHORT_HASH = repo.git.rev_parse(repo.active_branch.commit.hexsha, short=8)
    LONG_HASH = repo.active_branch.commit.hexsha
else:
    SHORT_HASH = os.getenv("CI_COMMIT_SHORT_SHA", "HEAD")
    LONG_HASH = os.getenv("CI_COMMIT_SHA", "HEAD")

# release = BRANCH_NAME + "@" + SHORT_HASH

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinxcontrib.katex",
    "sphinx_plotly_directive",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
    "sphinxarg.ext",
]
autosummary_generate = True
always_document_param_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "xarray": ("http://xarray.pydata.org/en/v0.19.0/", None),
    "numpy": ("https://numpy.org/doc/1.21/", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
}


# need these for intersphinx to work
import xarray

xarray.DataArray.__module__ = "xarray"

import plotly.graph_objs._figure

plotly.graph_objs._figure.Figure.__module__ = "plotly.graph_objects"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*test_*",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_show_sphinx = False
html_logo = "_static/assets/dolby_symbol.png"

html_theme_options = {"collapse_navigation": False}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "fonts/avenir-next-lt-pro/avenir-next-lt-pro.css",
    "fonts/FiraCode_2/fira_code.css",
    "css/extra.css",
]
html_js_files = [("js/mermaid.min.js", {"priority": 100})]

html_style = "css/dlb_theme.css"

html_favicon = "_static/assets/favicon.ico"

plotly_html_show_formats = False
plotly_html_show_source_link = False

plotly_iframe_width = "100%"
plotly_iframe_height = "400px"

mermaid_version = ""
mermaid_init_js = """mermaid.initialize({
    startOnLoad: true,
    flowchart: {curve: "linear", htmlLabels: true},
    theme: "base",
    securityLevel: "loose",
    themeVariables: {
        darkMode: true,
        fontFamily: "Avenir Next LT Pro Regular",
        background: "#f4f4f4",
        primaryColor: "#000",
        secondaryColor: "#FF2E7E",
        edgeLabelBackground: "#3b48fe",
        tertiaryColor: "#AA33FF",
        tertiaryTextColor: "#fff",
        defaultLinkColor: "#000",
        lineColor: "#000",
        primaryTextColor: "#fff",
        noteBkgColor: "#FF2E7E"
    },
    themeCSS: '.label { font-family: Avenir Next LT Pro Regular,Arial,sans-serif; }'
});"""

autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_default_options = {"undoc-members": True}
# autodoc_aliases = {"utennsil.ufb_banding": "ufb_banding"}

# https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py
# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(spatpy.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "%s#L%d-%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    return f"https://gitlab-sfo.dolby.net/capture/spatpy/-/blob/{LONG_HASH}/spatpy/{filename}"


APIDOC_DIR = os.path.join(DOC_DIR, "_apidoc")

import subprocess


def setup(app):
    def process_bases(app, name, obj, options, bases):
        if (
            dataclasses.is_dataclass(obj)
            and len(bases) == 1
            and bases[0].__name__ == "object"
        ):
            bases[0] = (
                "`@dataclass"
                " <https://docs.python.org/3/library/dataclasses.html>`_"
            )

    app.connect("autodoc-process-bases", process_bases)

    cmd = [
        "sphinx-apidoc",
        "-o",
        APIDOC_DIR,
        "--module-first",
        "--separate",
        "--force",
        "--templatedir",
        os.path.join(DOC_DIR, "apidoc-templates"),
        SPATPY_DIR,
    ] + EXCLUDE_PATTERNS
    print(" ".join(cmd))
    subprocess.call(cmd)
