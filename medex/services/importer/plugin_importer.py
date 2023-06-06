import importlib
import os
import importlib.util
import pathlib


class PluginImporter:
    def __init__(self, plugin_folder):
        self.plugin_folder = plugin_folder
        self.plugin_modules: list[PluginInterface] = []

    def import_plugins(self):
        plugin_files = self._get_plugin_files()

        for file in plugin_files:
            module_name = PluginImporter._get_module_name(file)
            plugin_module = self._import_module(module_name, file)
            plugin_module.on_loaded()

            if plugin_module:
                self.plugin_modules.append(plugin_module)


    def on_db_ready(self, session):
        for plugin_module in self.plugin_modules:
            plugin_module.on_db_ready(session)

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
            if issubclass(PluginInterface):
                return plugin_module
            else:
                print(f"Error importing module '{module_name}': Does not implement PluginInterface")
                return None

        except Exception as e:
            print(f"Error importing module '{module_name}': {e}")
            return None

