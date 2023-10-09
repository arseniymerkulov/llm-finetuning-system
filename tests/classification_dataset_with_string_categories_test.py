import pytest


from lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'classification-dataset-cancer-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'DATA_ANALYSIS',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'datatattle/covid-19-nlp-text-classification',
        'dataset_file': 'Corona_NLP_train.csv',
        'dataset_table_columns': ['OriginalTweet', 'Sentiment'],
        'dataset_partition': 1000,
        'model_alias': 'cointegrated/rubert-tiny2'
    })

    test_process.start_run()
    test_process.update_configuration('project')
    test_process.update_configuration('pipeline_setup')
    test_process.update_configuration('task')
    test_process.update_configuration('dataset_alias')
    test_process.update_configuration('dataset_file')
    test_process.update_configuration('dataset_table_columns')
    test_process.update_configuration('dataset_partition')
    test_process.update_configuration('model_alias')
