
class PluginInterface:
    def on_loaded(self):
        pass

    def on_db_ready(self, session):
        pass

    def on_stopped(self):
        pass

    # ... Expand with further lifecycle methods


