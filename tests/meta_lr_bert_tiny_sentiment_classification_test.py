import pytest


from .lib import TestProcess


@pytest.mark.meta_classification
@pytest.mark.tiny
def test():
    test_name = 'meta-lr-bert-classification'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'jp797498e/twitter-entity-sentiment-analysis',
        'dataset_file': 'twitter_training.csv',
        'dataset_table_columns': ['im getting on borderlands and i will murder you all ,', 'Positive'],
        'dataset_partition': 100,
        'dataset_need_balance': True,
        'model_alias': 'prajjwal1/bert-tiny'
    })

    test_process.selection({
        'additional_learning_rate': [1e-3, 1e-4]
    })
