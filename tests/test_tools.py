"""Tests for vLLM MCP Server tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vllm_mcp_server.tools.chat import handle_chat, handle_complete
from vllm_mcp_server.tools.models import list_models, get_model_info


class TestChatTool:
    """Tests for chat tool."""

    @pytest.mark.asyncio
    async def test_handle_chat_no_messages(self):
        """Test chat with no messages returns error."""
        result = await handle_chat({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "No messages" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_chat_invalid_message_format(self):
        """Test chat with invalid message format returns error."""
        result = await handle_chat({"messages": [{"invalid": "format"}]})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "role" in result[0].text or "content" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_chat_success(self):
        """Test successful chat completion."""
        mock_response = {
            "choices": [
                {"message": {"content": "Hello! How can I help you?"}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            }
        }

        with patch("vllm_mcp_server.tools.chat.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await handle_chat({
                "messages": [{"role": "user", "content": "Hello"}]
            })

            assert len(result) == 1
            assert "Hello! How can I help you?" in result[0].text
            assert "Tokens:" in result[0].text


class TestCompleteTool:
    """Tests for completion tool."""

    @pytest.mark.asyncio
    async def test_handle_complete_no_prompt(self):
        """Test completion with no prompt returns error."""
        result = await handle_complete({})
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "No prompt" in result[0].text

    @pytest.mark.asyncio
    async def test_handle_complete_success(self):
        """Test successful text completion."""
        mock_response = {
            "choices": [
                {"text": "world! This is a test completion."}
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            }
        }

        with patch("vllm_mcp_server.tools.chat.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.text_completion.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await handle_complete({"prompt": "Hello "})

            assert len(result) == 1
            assert "world!" in result[0].text


class TestModelTools:
    """Tests for model management tools."""

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Test list_models when no models available."""
        with patch("vllm_mcp_server.tools.models.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.list_models.return_value = []
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await list_models()

            assert len(result) == 1
            assert "No models" in result[0].text

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test list_models returns model list."""
        mock_models = [
            {"id": "model-1", "owned_by": "vllm", "created": 1234567890},
            {"id": "model-2", "owned_by": "vllm", "created": 1234567890},
        ]

        with patch("vllm_mcp_server.tools.models.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.list_models.return_value = mock_models
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await list_models()

            assert len(result) == 1
            assert "model-1" in result[0].text
            assert "model-2" in result[0].text
            assert "2 total" in result[0].text

    @pytest.mark.asyncio
    async def test_get_model_info_no_model_id(self):
        """Test get_model_info with no model_id returns error."""
        result = await get_model_info({})
        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_get_model_info_not_found(self):
        """Test get_model_info when model not found."""
        with patch("vllm_mcp_server.tools.models.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get_model_info.return_value = None
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_model_info({"model_id": "nonexistent"})

            assert len(result) == 1
            assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_get_model_info_success(self):
        """Test get_model_info returns model info."""
        mock_model = {"id": "test-model", "owned_by": "vllm", "created": 1234567890}

        with patch("vllm_mcp_server.tools.models.VLLMClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get_model_info.return_value = mock_model
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await get_model_info({"model_id": "test-model"})

            assert len(result) == 1
            assert "test-model" in result[0].text

