# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
APIDIR        = source/user_guide/api
CONTRIBDIR    = source/user_guide/contrib
SPHINXAPIDOC  = sphinx-apidoc
SPHINXAUTOGEN = sphinx-autogen

clean:
	@echo
	@echo "Cleaning up..."
	@echo
	rm -rf $(BUILDDIR)/html/*
	rm -rf $(BUILDDIR)/doctrees
	rm -rf $(BUILDDIR)/doctest
	rm -rf $(APIDIR)/generated
	rm -rf $(CONTRIBDIR)/generated
	rm -rf $(CONTRIBDIR)/index.rst

update_api: clean
	@echo
	@echo "Building API docs..."
	@echo
	$(SPHINXAUTOGEN) -i -t _templates -o $(BUILDDIR)/$(APIDIR)/generated $(SOURCEDIR)/$(APIDIR)/index.rst


html: clean
	@echo
	@echo "Building HTML files..."
	@echo
	$(SPHINXBUILD) -d $(BUILDDIR)/doctrees -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
