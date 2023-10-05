import pytest


from lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'causal-lm-finetuning-gpt2-medium-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'task': 'CAUSAL_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'model_alias': 'gpt2-medium'
    })

    TestProcess.start_run()
    test_process.update_configuration('project')
    test_process.update_configuration('task')
    test_process.update_configuration('dataset_alias')
    test_process.update_configuration('model_alias')
