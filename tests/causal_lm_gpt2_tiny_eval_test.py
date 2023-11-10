import pytest


from .lib import TestProcess


@pytest.mark.evaluation
@pytest.mark.tiny
def test():
    test_name = 'causal-lm-finetuning-gpt2-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'EVALUATION',
        'task': 'CAUSAL_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'dataset_file': 'news_summary.csv',
        'dataset_table_columns': ['text', 'ctext'],
        'dataset_partition': 1000,
        'model_alias': 'sshleifer/tiny-gpt2'
    })

    test_process.start_run()
    test_process.execute_pipeline()
