.PHONY: docs

docs: # requires `enchant` to be installed
	rm -rf build
	rm -rf docs/generated
	rm -rf docs/sites/generated
	sphinx-build -W -b html docs build
