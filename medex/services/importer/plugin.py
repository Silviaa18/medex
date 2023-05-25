import importlib
import os
import importlib.util
import pathlib


class PluginImporter:
    def __init__(self, plugin_folder):
        self.plugin_folder = plugin_folder

    def import_plugins(self):
        plugin_files = self._get_plugin_files()
        plugin_modules = []

        for file in plugin_files:
            module_name = PluginImporter._get_module_name(file)
            plugin_module = self._import_module(module_name, file)

            if plugin_module:
                plugin_modules.append(plugin_module)

        return plugin_modules

    def _get_plugin_files(self):
        if not os.path.exists(self.plugin_folder):
            return []

        folders = pathlib.Path(self.plugin_folder).glob('*/')
        plugin_files: list[str] = [str(plugin) for folder in folders for plugin in folder.glob('*.py')]
        return plugin_files

    @staticmethod
    def _get_module_name(file):
        module_name = os.path.splitext(file)[0]
        return module_name

    @staticmethod
    def _import_module(module_name, file):
        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            return plugin_module
        except Exception as e:
            print(f"Error importing module '{module_name}': {e}")
            return None
