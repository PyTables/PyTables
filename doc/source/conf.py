# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


def get_version(filename):
    import re

    with open(filename) as fd:
        data = fd.read()

    mobj = re.search(
        r"""^__version__\s*=\s*(?P<quote>['"])(?P<version>.*)(?P=quote)""",
        data,
        re.MULTILINE,
    )
    return mobj.group("version")


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyTables"
copyright = "2011–2025, PyTables maintainers"  # noqa: A001
author = "PyTables maintainers"
version = get_version("../../tables/_version.py")
release = version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

needs_sphinx = "1.3"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # 'numpydoc',
    "sphinx_rtd_theme",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    # 'sticky_navigation': True  # Set to False to disable the sticky nav while scrolling.
    "logo_only": True,  # if we have a html_logo below, this shows /only/ the logo with no title text
}
html_logo = "_static/logo-pytables-small.png"

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "PyTablesDoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""\usepackage{bookmark,hyperref}
\usepackage[para]{threeparttable}
\DeclareUnicodeCharacter{210F}{$\hbar$}""",
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (
        "usersguide/usersguide",
        f"usersguide-{version}.tex",
        "PyTables User Guide",
        "PyTables maintainers",
        "manual",
    ),
]

latex_logo = "usersguide/images/pytables-front-logo.pdf"
latex_use_parts = True
latex_domain_indices = False


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/": None}

# -- External link options ----------------------------------------------------
extlinks = {
    "issue": ("https://github.com/PyTables/PyTables/issues/%s", "gh-%s"),
    "PR": ("https://github.com/PyTables/PyTables/pull/%s", "gh-%s"),
    "commit": ("https://github.com/PyTables/PyTables/commit/%s", "commit %s"),
}

# -- Options for autodocumentation ---------------------------------------------
autodoc_member_order = "groupwise"
autoclass_content = "class"
autosummary_generate = []
