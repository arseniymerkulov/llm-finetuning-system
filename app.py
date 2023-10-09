from flask import Flask, request
from enum import Enum


from core.configuration.configuration import Configuration
from core.configuration.hyperparams import PipelineSetup
from core.main_process.main_process import MainProcess


app = Flask(__name__)


# todo: improve assert msgs
# todo: add data check pipeline setup
@app.route('/api/start', methods=['POST'])
def start_main_process():
    try:
        config = Configuration.get_instance()

        assert 'pipeline_setup' in request.json.keys(), 'request missing "pipeline_setup" field'
        value = request.json['pipeline_setup']

        assert isinstance(value, str), 'request field "pipeline_setup" is not <str> type'
        assert value.upper() in PipelineSetup.__members__, f'invalid value for enum field "PipelineSetup"'

        value = PipelineSetup[value.upper()]

        config.configure('pipeline_setup', value)

        thread = MainProcess(value)
        thread.daemon = True
        thread.start()

        return {'message': f'finetuning process started with pipeline setup "{value.name}"'}

    except AssertionError as e:
        return {'error': str(e)}


@app.route('/api/update', methods=['POST'])
def update_configuration():
    try:
        config = Configuration.get_instance()

        assert 'field' in request.json.keys(), 'request missing "field" field'
        assert 'value' in request.json.keys(), 'request missing "value" field'

        field = request.json['field']
        value = request.json['value']

        assert isinstance(field, str), 'request field "field" is not <str> type'
        assert hasattr(config, field), f'there is no such field "{field}" in the configuration'
        assert getattr(config, field) is not None, 'configuring NoneType fields from API is forbidden'
        desired_class = getattr(config, field).__class__

        if issubclass(desired_class, Enum):
            assert isinstance(value, str), 'request field "value" is not <str> type'
            assert value.upper() in desired_class.__members__, f'invalid "value" for enum field "{field}"'

            value = desired_class[value.upper()]

        else:
            assert isinstance(value, desired_class), \
                f'request field "value" is not {desired_class} type'

        config.configure(field, value)
        return {'message': 'configuration updated'}

    except AssertionError as e:
        return {'error': str(e)}
