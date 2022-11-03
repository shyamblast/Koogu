
# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Koogu'
copyright = '2020–present, Shyam Madhusudhana'
author = 'Shyam Madhusudhana'

# The first 2 components of software version
version = '0.7.1'
# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'recommonmark'
]

source_suffix = ['.rst']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

master_doc = 'index'

language = 'en'
locale_dirs = ['locale/']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

epub_basename = 'target'

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_show_sourcelink = False

try:
    html_context
except NameError:
    html_context = {}
html_context['display_lower_left'] = True

if 'REPO_NAME' in os.environ:
    REPO_NAME = os.environ['REPO_NAME']
else:
    REPO_NAME = project
 
from git import Repo
repo = Repo(search_parent_directories=True)

if 'current_version' in os.environ:
    # get the current_version env var set by build_docs.sh
    current_version = os.environ['current_version']
else:
    # the user is probably doing `make html`
    # set this build's current version by looking at the branch
    current_version = repo.active_branch.name

# rename the 'master' bracnh to be version = 'stable'
if current_version == 'master':
    current_version = 'stable'

# tell the theme which version we're currently on ('current_version' affects
# the lower-left rtd menu and 'version' affects the logo-area version)
html_context['current_version'] = current_version
html_context['version'] = current_version

if 'current_language' in os.environ:
    # get the current_language env var set by build_docs.sh
    current_language = os.environ['current_language']
else:
    # the user is probably doing `make html`
    # set this build's current language to english
    current_language = 'en'

html_context['current_language'] = current_language

# POPULATE LINKS TO OTHER LANGUAGES
html_context['languages'] = [ ('en', '/{}/en/{}/'.format(REPO_NAME, current_version)) ]

languages = [lang.name for lang in os.scandir('locale') if lang.is_dir()]
for lang in languages:
    html_context['languages'].append( (lang, '/{}/{}/{}/'.format(REPO_NAME, lang, current_version)) )

# POPULATE LINKS TO OTHER VERSIONS
html_context['versions'] = list()

# get list of remote branches, excluding HEAD and gh-pages
main_versions = list()
branch_versions = list()
for ref in repo.remote().refs:
    ref = ref.name.split('/')[-1]
    if ref == 'master':
        # special override to rename 'master' branch to 'stable'
        main_versions.append('stable')
    elif ref == 'dev':
        main_versions.append(ref)
    elif ref != 'HEAD' and ref != 'gh-pages':
        branch_versions.append(ref)

for ver in (main_versions + sorted(branch_versions)[::-1]):
    html_context['versions'].append(
        (ver, '/{}/{}/{}/'.format(REPO_NAME, current_language, ver))
    )

# POPULATE LINKS TO OTHER FORMATS/DOWNLOADS
 
# settings for creating PDF with rinoh
rinoh_documents = [(
    master_doc,
    'target',
    '{} Documentation'.format(REPO_NAME),
    '© {}'.format(copyright),
)]
today_fmt = "%B %d, %Y"

download_filepath = '/{}/{}/{}/{}-docs_{}_{}'.format(
    REPO_NAME, current_language, current_version, REPO_NAME, current_language, current_version)

html_context['downloads'] = list()
#html_context['downloads'].append( ('pdf', download_filepath + '.pdf') )    # disable for now since rinoh is failing
html_context['downloads'].append( ('epub', download_filepath + '.epub') )
 


# -- Autodoc options ---------------------------------------------------------
autodoc_member_order = 'groupwise'
autodoc_mock_imports = [
    #'numpy',
    'scipy',
    'soundfile',
    'audioread',
    'resampy',
    'tensorflow',
]

# -- Code highlight options --------------------------------------------------
highlight_language = 'python3'

