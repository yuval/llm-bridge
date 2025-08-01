import sys
import pathlib
import pytest

# Ensure the src directory is on the path for importing llm_bridge modules directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from llm_bridge.responses import OpenAIResponse
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as ToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice

def test_invalid_tool_call_arguments_returns_empty_dict_instead_of_none():
    bad_json = '{not valid json'

    function = ToolCallFunction(name='test', arguments=bad_json)
    tool_call = ChatCompletionMessageToolCall(id='id1', type='function', function=function)
    message = ChatCompletionMessage(role='assistant', tool_calls=[tool_call])
    choice = Choice(finish_reason='stop', index=0, message=message)
    completion = ChatCompletion(
        id='cmpl-1',
        choices=[choice],
        created=0,
        model='gpt-4',
        object='chat.completion',
    )

    response = OpenAIResponse(raw_response=completion)
    tool_calls = response.get_tool_calls()

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == 'id1'
    assert tool_calls[0].name == 'test'
    assert tool_calls[0].arguments == {}
