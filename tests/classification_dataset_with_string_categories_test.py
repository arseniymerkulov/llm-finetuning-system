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

    test_process.start_run()

    test_process.set_stage('EnvironmentCleaning')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('EnvironmentSetup')
    test_process.update_configuration('project')
    test_process.update_configuration('pipeline_setup')
    test_process.update_configuration('task')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('DataCollecting')
    test_process.update_configuration('dataset_alias')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('DataProcessing')
    test_process.update_configuration('dataset_file')
    test_process.update_configuration('dataset_table_columns')
    test_process.update_configuration('dataset_partition')
    test_process.update_configuration('dataset_balance')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('ModelSelection')
    test_process.update_configuration('model_alias')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('DataTokenizing')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('FinetuningMethodSelection')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('HPO')
    test_process.wait('approve')
    test_process.approve_stage()

    test_process.set_stage('EnvironmentCleaning')
    test_process.wait('approve')
    test_process.approve_stage()
