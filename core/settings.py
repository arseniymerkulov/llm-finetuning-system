

class Settings:
    _instance = None

    def __init__(self):
        self.app_endpoint = 'http://127.0.0.1:5000'
        self.projects_dir = 'data/projects'
        self.datasets_dir = 'data/datasets'
        self.checkpoint_postfix = '-{epoch:02d}-{val_loss:.2f}'

        self.field_configure_timeout = 100
        self.stage_approval_timeout = 100

    @staticmethod
    def get_instance():
        if not Settings._instance:
            Settings._instance = Settings()

        return Settings._instance
