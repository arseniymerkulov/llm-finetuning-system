import pytest


from .lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'classification-dataset-with-string-categories-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'DATA_ANALYSIS',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'datatattle/covid-19-nlp-text-classification',
        'dataset_file': 'Corona_NLP_train.csv',
        'dataset_table_columns': ['OriginalTweet', 'Sentiment'],
        'dataset_partition': 1000,
        'dataset_balance': True,
        'model_alias': 'cointegrated/rubert-tiny2'
    })

    test_process.run_pipeline()
