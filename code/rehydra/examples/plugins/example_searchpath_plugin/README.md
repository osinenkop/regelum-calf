# Rehydra example SearchPath plugin

This plugin provides an example for how to write a SearchPathPlugin that can manipulate the search path.
Typical use cases includes:
 * A framework that wants to allow user code to discover its configurations and be able to compose with them.
 * A plugin that wants to extend Rehydra or another plugin by providing additional configs in existing config groups like `rehydra/launcher`.
 * A plugin that can replace the default configuration of another plugin or of Rehydra itself by prepending its configurations before those that it want to replace.
 
SearchPath plugins are discovered and enabled automatically once they are installed.
You can use `python foo.py --info` to see the search path and the installed plugins.
