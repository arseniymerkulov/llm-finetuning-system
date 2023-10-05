import wandb
import json
import os


from core.main_process.pipeline import Stage


class EnvironmentSetup(Stage):
    def execute(self):
        self.config.wait('project')
        self.config.configure('project_dir', f'{self.settings.projects_dir}/{self.config.project}')
        self.config.wait('task')

        self.config.configure('wandb_token_path', 'data/wandb.json')
        wandb.login(key=self._load_wandb_token())

        if not os.path.exists(self.config.project_dir):
            os.mkdir(self.config.project_dir)

    def _load_wandb_token(self):
        with open(self.config.wandb_token_path, 'r') as file:
            return json.loads(file.read())['key']

    def validate(self) -> bool:
        return os.path.exists(self.config.project_dir)
