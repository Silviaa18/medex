import importlib
import unittest
from unittest import mock

from pyfakefs.fake_filesystem_unittest import TestCase

from medex.services.importer.plugin import PluginImporter


class PluginImporterTest(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_import_plugins_no_files(self):
        importer = PluginImporter('plugins')
        with mock.patch('importlib.util.spec_from_file_location') as mock_spec_from_file_location:
            def mock_spec(module_name, file):
                plugin_module = importlib.util.module_from_spec(mock_spec_from_file_location.return_value)
                mock_spec_from_file_location.return_value.loader.exec_module(plugin_module)
                return plugin_module

            mock_spec_from_file_location.side_effect = mock_spec
        plugins = importer.import_plugins()
        self.assertEqual(len(plugins), 0)

    def test_import_plugins_with_files(self):
        self.fs.create_file('plugins/plugin1/plugin1.py')
        self.fs.create_file('plugins/plugin2/plugin2.py')
        importer = PluginImporter(plugin_folder="plugins")
        with mock.patch('importlib.util.spec_from_file_location') as mock_spec_from_file_location:
            def mock_spec(module_name, file):
                plugin_module = importlib.util.module_from_spec(mock_spec_from_file_location.return_value)
                mock_spec_from_file_location.return_value.loader.exec_module(plugin_module)
                return plugin_module

            mock_spec_from_file_location.side_effect = mock_spec
            plugins = importer.import_plugins()
        self.assertEqual(len(plugins), 2)

    def test_import_plugins_with_invalid_module(self):
        self.fs.create_file('plugins/plugin1/invalid_plugin.py')

        importer = PluginImporter('plugins')
        with mock.patch('importlib.util.spec_from_file_location') as mock_spec_from_file_location:
            def mock_spec(module_name, file):
                plugin_module = importlib.util.module_from_spec(mock_spec_from_file_location.return_value)
                mock_spec_from_file_location.return_value.loader.exec_module(plugin_module)
                return plugin_module

            mock_spec_from_file_location.side_effect = mock_spec
        plugins = importer.import_plugins()

        self.assertEqual(len(plugins), 0)

    def test_import_plugins_folder_not_exists(self):
        importer = PluginImporter('plugins')
        with mock.patch('importlib.util.spec_from_file_location') as mock_spec_from_file_location:
            def mock_spec(module_name, file):
                plugin_module = importlib.util.module_from_spec(mock_spec_from_file_location.return_value)
                mock_spec_from_file_location.return_value.loader.exec_module(plugin_module)
                return plugin_module

            mock_spec_from_file_location.side_effect = mock_spec
        plugins = importer.import_plugins()

        self.assertEqual(len(plugins), 0)


if __name__ == '__main__':
    unittest.main()
