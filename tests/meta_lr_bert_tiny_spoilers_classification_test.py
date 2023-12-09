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
        'dataset_alias': 'rmisra/imdb-spoiler-dataset',
        'dataset_file': 'IMDB_reviews.json',
        'dataset_table_columns': ['review_text', 'is_spoiler'],
        'dataset_partition': 100000,
        'dataset_need_balance': True,
        'model_alias': 'prajjwal1/bert-tiny'
    })

    test_process.selection({
        'additional_learning_rate': [1e-3, 1e-4]
    })
