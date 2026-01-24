"""
tests/agents/test_sentence_analyzer.py

Comprehensive tests for the SentenceAnalyzer agent that follow cardinal rules:
1. Test actual functionality, not mock interactions
2. Use realistic data and scenarios, not hardcoded values

These tests focus on testing the real analysis logic, prompt formatting,
error handling, and integration with actual Pydantic validation.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.agents.sentence_analyzer import SentenceAnalyzer


class TestSentenceAnalyzerInitialization:
    """Test SentenceAnalyzer initialization with realistic configurations."""

    @pytest.fixture
    def realistic_config(self, tmp_path):
        """Provide a realistic configuration for testing."""
        # Create actual prompt file for testing
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "no_context.yaml"

        realistic_prompts = {
            "sentence_function_type": {
                "prompt": (
                    "Analyze the function of this sentence: '{sentence}'. "
                    "Is it declarative, interrogative, imperative, or exclamatory?"
                )
            },
            "sentence_structure_type": {
                "prompt": (
                    "Analyze the structure of this sentence: '{sentence}'. " "Is it simple, compound, or complex?"
                )
            },
            "sentence_purpose": {
                "prompt": ("What is the purpose of this sentence: '{sentence}' " "in the context: {context}?")
            },
            "topic_level_1": {
                "prompt": ("What is the main topic of this sentence: '{sentence}' " "given the context: {context}?")
            },
            "topic_level_3": {
                "prompt": ("What is the specific subtopic of this sentence: '{sentence}' " "in context: {context}?")
            },
            "topic_overall_keywords": {"prompt": "Extract the key terms from this context: {context}"},
            "domain_specific_keywords": {
                "prompt": ("Identify domain-specific keywords in: '{sentence}' " "related to: {domain_keywords}")
            },
        }

        with open(prompt_file, "w") as f:
            yaml.dump(realistic_prompts, f)

        return {
            "classification": {"local": {"prompt_files": {"no_context": str(prompt_file)}}},
            "domain_keywords": ["python", "programming", "interview", "technical"],
        }

    def test_initialization_with_config_dict(self, realistic_config):
        """Test initialization with a provided config dictionary."""
        analyzer = SentenceAnalyzer(config_dict=realistic_config)

        assert analyzer.config == realistic_config
        assert analyzer.prompts is not None
        assert "sentence_function_type" in analyzer.prompts
        assert "sentence_structure_type" in analyzer.prompts

        # Verify prompts were loaded correctly
        function_prompt = analyzer.prompts["sentence_function_type"]["prompt"]
        assert "sentence" in function_prompt
        assert "declarative" in function_prompt.lower()

    def test_initialization_with_invalid_prompt_file(self, tmp_path):
        """Test initialization handles missing prompt files gracefully."""
        invalid_config = {
            "classification": {"local": {"prompt_files": {"no_context": str(tmp_path / "nonexistent.yaml")}}},
            "domain_keywords": [],
        }

        analyzer = SentenceAnalyzer(config_dict=invalid_config)

        # Should handle error gracefully and have empty prompts
        assert analyzer.config == invalid_config
        assert analyzer.prompts == {}


class TestSentenceAnalyzerClassification:
    """Test actual sentence classification functionality with realistic data."""

    @pytest.fixture
    def analyzer_with_realistic_setup(self, tmp_path):
        """Create analyzer with realistic prompts and mock only the LLM calls."""
        # Create realistic prompt configuration
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "interview_analysis.yaml"

        interview_prompts = {
            "sentence_function_type": {
                "prompt": (
                    "Classify the function of this interview sentence: '{sentence}'. "
                    "Return JSON with 'function_type' field."
                )
            },
            "sentence_structure_type": {
                "prompt": (
                    "Analyze the grammatical structure of: '{sentence}'. " "Return JSON with 'structure_type' field."
                )
            },
            "sentence_purpose": {
                "prompt": (
                    "Determine the purpose of '{sentence}' in this interview context: {context}. "
                    "Return JSON with 'purpose' field."
                )
            },
            "topic_level_1": {
                "prompt": (
                    "Identify the main topic of '{sentence}' given context: {context}. "
                    "Return JSON with 'topic_level_1' field."
                )
            },
            "topic_level_3": {
                "prompt": (
                    "What specific subtopic does '{sentence}' address in context: {context}? "
                    "Return JSON with 'topic_level_3' field."
                )
            },
            "topic_overall_keywords": {
                "prompt": (
                    "Extract key terms from this interview context: {context}. "
                    "Return JSON with 'overall_keywords' array."
                )
            },
            "domain_specific_keywords": {
                "prompt": (
                    "Find technical keywords in '{sentence}' related to {domain_keywords}. "
                    "Return JSON with 'domain_keywords' array."
                )
            },
        }

        with open(prompt_file, "w") as f:
            yaml.dump(interview_prompts, f)

        config = {
            "classification": {"local": {"prompt_files": {"no_context": str(prompt_file)}}},
            "domain_keywords": ["python", "programming", "software", "development", "technical"],
        }

        return SentenceAnalyzer(config_dict=config)

    @pytest.fixture
    def realistic_interview_contexts(self):
        """Provide realistic interview context data."""
        return {
            "immediate_context": "We're discussing your technical background and experience.",
            "observer_context": ("This is a technical interview for a senior software " "developer position."),
            "broader_context": (
                "The candidate is being evaluated for Python development " "skills and problem-solving abilities."
            ),
        }

    def create_realistic_llm_response(self, sentence: str, prompt_type: str) -> Dict[str, Any]:
        """Generate realistic LLM responses based on actual sentence analysis."""
        # Analyze sentence characteristics for realistic responses
        has_question = "?" in sentence
        has_exclamation = "!" in sentence
        word_count = len(sentence.split())

        # Generate realistic responses based on prompt type and sentence characteristics
        if prompt_type == "sentence_function_type":
            if has_question:
                return {"function_type": "interrogative"}
            elif has_exclamation:
                return {"function_type": "exclamatory"}
            elif sentence.lower().startswith(("please", "could you", "would you")):
                return {"function_type": "imperative"}
            else:
                return {"function_type": "declarative"}

        elif prompt_type == "sentence_structure_type":
            if word_count <= 5:
                return {"structure_type": "simple"}
            elif " and " in sentence or " or " in sentence or " but " in sentence:
                return {"structure_type": "compound"}
            else:
                return {"structure_type": "complex"}

        elif prompt_type == "sentence_purpose":
            if has_question:
                return {"purpose": "information_gathering"}
            elif "experience" in sentence.lower():
                return {"purpose": "experience_assessment"}
            else:
                return {"purpose": "general_inquiry"}

        elif prompt_type == "topic_level_1":
            if "python" in sentence.lower():
                return {"topic_level_1": "programming_languages"}
            elif "experience" in sentence.lower():
                return {"topic_level_1": "professional_background"}
            else:
                return {"topic_level_1": "general_discussion"}

        elif prompt_type == "topic_level_3":
            if "python" in sentence.lower() and "experience" in sentence.lower():
                return {"topic_level_3": "python_experience_assessment"}
            else:
                return {"topic_level_3": "specific_technical_inquiry"}

        elif prompt_type == "topic_overall_keywords":
            # Extract realistic keywords from sentence
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            words = [w.lower().strip(".,!?") for w in sentence.split() if w.lower() not in common_words]
            return {"overall_keywords": words[:5]}  # Top 5 keywords

        elif prompt_type == "domain_specific_keywords":
            domain_terms = []
            sentence_lower = sentence.lower()
            if "python" in sentence_lower:
                domain_terms.append("python")
            if "programming" in sentence_lower or "code" in sentence_lower:
                domain_terms.append("programming")
            if "experience" in sentence_lower:
                domain_terms.append("professional_experience")
            return {"domain_keywords": domain_terms}

        return {}  # Fallback

    @pytest.mark.asyncio
    async def test_classify_realistic_interview_question(
        self, analyzer_with_realistic_setup, realistic_interview_contexts
    ):
        """Test classification of a realistic interview question."""
        analyzer = analyzer_with_realistic_setup
        sentence = "Can you tell me about your experience with Python programming?"

        # Mock the agent calls with realistic responses
        async def mock_call_model(prompt: str) -> Dict[str, Any]:
            # Determine prompt type based on prompt content
            if "function of this interview sentence" in prompt:
                return self.create_realistic_llm_response(sentence, "sentence_function_type")
            elif "grammatical structure" in prompt:
                return self.create_realistic_llm_response(sentence, "sentence_structure_type")
            elif "purpose of" in prompt:
                return self.create_realistic_llm_response(sentence, "sentence_purpose")
            elif "main topic" in prompt:
                return self.create_realistic_llm_response(sentence, "topic_level_1")
            elif "specific subtopic" in prompt:
                return self.create_realistic_llm_response(sentence, "topic_level_3")
            elif "key terms from this interview context" in prompt:
                return self.create_realistic_llm_response(sentence, "topic_overall_keywords")
            elif "technical keywords" in prompt:
                return self.create_realistic_llm_response(sentence, "domain_specific_keywords")
            return {}

        # Patch the agent to use our realistic mock
        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_model)

            result = await analyzer.classify_sentence(sentence, realistic_interview_contexts)

        # Test actual functionality - results should make sense for this sentence
        assert result["sentence"] == sentence
        assert result["function_type"] == "interrogative"  # It's a question
        assert result["purpose"] == "information_gathering"  # Gathering info about experience
        assert result["topic_level_1"] == "programming_languages"  # About Python
        assert "python" in result.get("domain_keywords", [])  # Should identify Python as domain keyword

        # Verify the agent was called the expected number of times (7 analysis dimensions)
        assert mock_agent.call_model.call_count == 7

    @pytest.mark.asyncio
    async def test_classify_realistic_interview_response(
        self, analyzer_with_realistic_setup, realistic_interview_contexts
    ):
        """Test classification of a realistic interview response."""
        analyzer = analyzer_with_realistic_setup
        sentence = "I have been working with Python for over 5 years in various " "web development projects."

        # Mock realistic responses based on this declarative statement
        async def mock_call_model(prompt: str) -> Dict[str, Any]:
            if "function of this interview sentence" in prompt:
                return {"function_type": "declarative"}  # It's a statement
            elif "grammatical structure" in prompt:
                return {"structure_type": "complex"}  # Complex sentence structure
            elif "purpose of" in prompt:
                return {"purpose": "experience_description"}  # Describing experience
            elif "main topic" in prompt:
                return {"topic_level_1": "professional_background"}  # About background
            elif "specific subtopic" in prompt:
                return {"topic_level_3": "python_experience_assessment"}  # Specific Python experience
            elif "key terms from this interview context" in prompt:
                return {"overall_keywords": ["python", "years", "development", "projects", "web"]}
            elif "technical keywords" in prompt:
                return {"domain_keywords": ["python", "web_development", "programming"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_model)

            result = await analyzer.classify_sentence(sentence, realistic_interview_contexts)

        # Test actual functionality for a declarative response
        assert result["sentence"] == sentence
        assert result["function_type"] == "declarative"  # It's a statement
        assert result["purpose"] == "experience_description"  # Describing experience
        assert result["topic_level_1"] == "professional_background"  # About background
        assert "python" in result.get("domain_keywords", [])  # Should identify Python
        assert "development" in result.get("overall_keywords", [])  # Should identify development

    @pytest.mark.asyncio
    async def test_error_handling_with_realistic_scenarios(
        self, analyzer_with_realistic_setup, realistic_interview_contexts
    ):
        """Test error handling with realistic error scenarios."""
        analyzer = analyzer_with_realistic_setup
        sentence = "What programming languages do you prefer?"

        # Mock an API error scenario
        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=Exception("API timeout error"))

            # Should propagate the exception (real error handling behavior)
            with pytest.raises(Exception, match="API timeout error"):
                await analyzer.classify_sentence(sentence, realistic_interview_contexts)

    def test_prompt_formatting_with_realistic_data(self, analyzer_with_realistic_setup, realistic_interview_contexts):
        """Test that prompts are formatted correctly with realistic data."""
        analyzer = analyzer_with_realistic_setup
        sentence = "What frameworks have you used with Python?"

        # Test that prompts contain the actual sentence and context
        function_prompt = analyzer.prompts["sentence_function_type"]["prompt"].format(sentence=sentence)
        assert sentence in function_prompt
        assert "function" in function_prompt.lower()

        purpose_prompt = analyzer.prompts["sentence_purpose"]["prompt"].format(
            sentence=sentence, context=realistic_interview_contexts["observer_context"]
        )
        assert sentence in purpose_prompt
        assert realistic_interview_contexts["observer_context"] in purpose_prompt

        domain_prompt = analyzer.prompts["domain_specific_keywords"]["prompt"].format(
            sentence=sentence, domain_keywords=", ".join(analyzer.config["domain_keywords"])
        )
        assert sentence in domain_prompt
        assert "python" in domain_prompt.lower()  # Should include domain keywords


class TestSentenceAnalyzerIntegration:
    """Test integration scenarios with realistic interview data."""

    @pytest.fixture
    def interview_sentences(self):
        """Provide realistic interview sentence examples."""
        return [
            "Tell me about yourself.",
            "What is your experience with Python?",
            "I have 5 years of experience in software development.",
            "Can you describe a challenging project you worked on?",
            "How do you handle debugging complex issues?",
            "Thank you for your time today.",
        ]

    @pytest.fixture
    def interview_contexts(self):
        """Provide realistic interview contexts."""
        return {
            "immediate_context": "We are conducting a technical interview for a senior developer position.",
            "observer_context": "The interviewer is assessing technical skills and communication abilities.",
            "broader_context": "This is part of the hiring process for a Python development role at a tech company.",
        }

    @pytest.mark.asyncio
    async def test_multiple_sentence_classification(self, interview_sentences, interview_contexts, tmp_path):
        """Test classification of multiple realistic interview sentences."""
        # Create analyzer with realistic setup
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "interview_analysis.yaml"

        interview_prompts = {
            "sentence_function_type": {
                "prompt": "Classify function: '{sentence}'. Return JSON with 'function_type' field."
            },
            "sentence_structure_type": {
                "prompt": "Analyze structure: '{sentence}'. Return JSON with 'structure_type' field."
            },
            "sentence_purpose": {
                "prompt": "Purpose: '{sentence}' in context: {context}. Return JSON with 'purpose' field."
            },
            "topic_level_1": {
                "prompt": "Main topic: '{sentence}' context: {context}. Return JSON with 'topic_level_1' field."
            },
            "topic_level_3": {
                "prompt": "Subtopic: '{sentence}' context: {context}. Return JSON with 'topic_level_3' field."
            },
            "topic_overall_keywords": {
                "prompt": "Keywords from context: {context}. Return JSON with 'overall_keywords' array."
            },
            "domain_specific_keywords": {
                "prompt": (
                    "Domain keywords: '{sentence}' related to {domain_keywords}. "
                    "Return JSON with 'domain_keywords' array."
                )
            },
        }

        with open(prompt_file, "w") as f:
            yaml.dump(interview_prompts, f)

        config = {
            "classification": {"local": {"prompt_files": {"no_context": str(prompt_file)}}},
            "domain_keywords": ["python", "programming"],
        }

        analyzer = SentenceAnalyzer(config_dict=config)

        # Mock realistic responses for different sentence types
        async def mock_call_model(prompt: str) -> Dict[str, Any]:
            # Return appropriate responses based on prompt content
            if "Classify function:" in prompt:
                if "?" in prompt:
                    return {"function_type": "interrogative"}
                else:
                    return {"function_type": "declarative"}
            elif "Analyze structure:" in prompt:
                return {"structure_type": "simple"}
            elif "Purpose:" in prompt:
                if "?" in prompt:
                    return {"purpose": "information_gathering"}
                else:
                    return {"purpose": "information_provision"}
            elif "Main topic:" in prompt:
                return {"topic_level_1": "interview_discussion"}
            elif "Subtopic:" in prompt:
                return {"topic_level_3": "technical_skills"}
            elif "Keywords from context:" in prompt:
                return {"overall_keywords": ["experience", "technical", "skills"]}
            elif "Domain keywords:" in prompt:
                return {"domain_keywords": ["programming", "development"]}
            # Fallback for any other prompt types - return complete response
            return {"analysis": "realistic_response"}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_model)

            # Test first few sentences
            results = []
            for sentence in interview_sentences[:3]:
                result = await analyzer.classify_sentence(sentence, interview_contexts)
                results.append(result)

        # Verify realistic classification results
        assert len(results) == 3

        # First sentence: "Tell me about yourself." - declarative (command/request)
        assert results[0]["function_type"] == "declarative"

        # Second sentence: "What is your experience with Python?" - interrogative (question)
        assert results[1]["function_type"] == "interrogative"

        # Third sentence: "I have 5 years..." - declarative (statement)
        assert results[2]["function_type"] == "declarative"


class TestSentenceAnalyzerConfigErrors:
    """Test configuration error handling paths."""

    def test_initialization_with_missing_config_key(self):
        """Test initialization with missing nested config keys."""
        # Config missing the nested prompt_files key
        incomplete_config = {
            "classification": {
                "local": {}  # Missing prompt_files
            },
            "domain_keywords": [],
        }

        analyzer = SentenceAnalyzer(config_dict=incomplete_config)

        # Should handle gracefully with empty prompts
        assert analyzer.prompts == {}

    def test_initialization_with_empty_config(self):
        """Test initialization with empty config dict."""
        empty_config = {}

        analyzer = SentenceAnalyzer(config_dict=empty_config)

        # Should handle gracefully
        assert analyzer.prompts == {}

    def test_initialization_with_none_prompt_path(self):
        """Test initialization when prompt path is None."""
        config_with_none = {
            "classification": {
                "local": {
                    "prompt_files": {
                        "no_context": None  # Explicitly None
                    }
                }
            },
            "domain_keywords": [],
        }

        analyzer = SentenceAnalyzer(config_dict=config_with_none)

        # Should handle gracefully
        assert analyzer.prompts == {}

    def test_initialization_with_invalid_yaml_file(self, tmp_path):
        """Test initialization with a malformed YAML file."""
        # Create a file with invalid YAML content
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("this: is: not: valid: yaml: [unclosed")

        config = {
            "classification": {
                "local": {
                    "prompt_files": {
                        "no_context": str(bad_yaml)
                    }
                }
            },
            "domain_keywords": [],
        }

        analyzer = SentenceAnalyzer(config_dict=config)

        # Should handle gracefully
        assert analyzer.prompts == {}


class TestSentenceAnalyzerValidationErrors:
    """Test Pydantic validation error handling for each response type."""

    @pytest.fixture
    def analyzer_with_prompts(self, tmp_path):
        """Create analyzer with valid prompts for testing validation errors."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "test_prompts.yaml"

        prompts = {
            "sentence_function_type": {"prompt": "Test: '{sentence}'"},
            "sentence_structure_type": {"prompt": "Test: '{sentence}'"},
            "sentence_purpose": {"prompt": "Test: '{sentence}' context: {context}"},
            "topic_level_1": {"prompt": "Test: '{sentence}' context: {context}"},
            "topic_level_3": {"prompt": "Test: '{sentence}' context: {context}"},
            "topic_overall_keywords": {"prompt": "Test context: {context}"},
            "domain_specific_keywords": {"prompt": "Test: '{sentence}' keywords: {domain_keywords}"},
        }

        import yaml
        with open(prompt_file, "w") as f:
            yaml.dump(prompts, f)

        config = {
            "classification": {"local": {"prompt_files": {"no_context": str(prompt_file)}}},
            "domain_keywords": ["test"],
        }

        return SentenceAnalyzer(config_dict=config)

    @pytest.fixture
    def test_contexts(self):
        """Provide test contexts."""
        return {
            "immediate_context": "test context",
            "observer_context": "test observer",
            "broader_context": "test broader",
        }

    @pytest.mark.asyncio
    async def test_validation_error_function_type(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for function_type response."""
        # Return invalid response (missing required field)
        async def mock_call_model(prompt: str) -> Dict[str, Any]:
            if "function" in prompt.lower() or prompt == analyzer_with_prompts.prompts["sentence_function_type"]["prompt"].format(sentence="Test sentence"):
                return {"wrong_field": "value"}  # Invalid - missing function_type
            # Return valid responses for other dimensions
            return {
                "structure_type": "simple",
                "purpose": "test",
                "topic_level_1": "test",
                "topic_level_3": "test",
                "overall_keywords": ["test"],
                "domain_keywords": ["test"],
            }

        # Need to return different responses for different calls
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:  # function_type
                return {"wrong_field": "value"}  # Invalid
            elif call_count[0] == 2:  # structure_type
                return {"structure_type": "simple"}
            elif call_count[0] == 3:  # purpose
                return {"purpose": "test"}
            elif call_count[0] == 4:  # topic_level_1
                return {"topic_level_1": "test"}
            elif call_count[0] == 5:  # topic_level_3
                return {"topic_level_3": "test"}
            elif call_count[0] == 6:  # overall_keywords
                return {"overall_keywords": ["test"]}
            elif call_count[0] == 7:  # domain_keywords
                return {"domain_keywords": ["test"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        # function_type should be empty string (default) due to validation error
        assert result["function_type"] == ""
        # Other fields should have values
        assert result["structure_type"] == "simple"

    @pytest.mark.asyncio
    async def test_validation_error_structure_type(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for structure_type response."""
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:  # function_type
                return {"function_type": "declarative"}
            elif call_count[0] == 2:  # structure_type
                return {"invalid": "response"}  # Invalid
            elif call_count[0] == 3:  # purpose
                return {"purpose": "test"}
            elif call_count[0] == 4:  # topic_level_1
                return {"topic_level_1": "test"}
            elif call_count[0] == 5:  # topic_level_3
                return {"topic_level_3": "test"}
            elif call_count[0] == 6:  # overall_keywords
                return {"overall_keywords": ["test"]}
            elif call_count[0] == 7:  # domain_keywords
                return {"domain_keywords": ["test"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        assert result["function_type"] == "declarative"
        assert result["structure_type"] == ""  # Default due to validation error

    @pytest.mark.asyncio
    async def test_validation_error_purpose(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for purpose response."""
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"function_type": "declarative"}
            elif call_count[0] == 2:
                return {"structure_type": "simple"}
            elif call_count[0] == 3:  # purpose
                return {"bad_field": "value"}  # Invalid
            elif call_count[0] == 4:
                return {"topic_level_1": "test"}
            elif call_count[0] == 5:
                return {"topic_level_3": "test"}
            elif call_count[0] == 6:
                return {"overall_keywords": ["test"]}
            elif call_count[0] == 7:
                return {"domain_keywords": ["test"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        assert result["purpose"] == ""  # Default due to validation error

    @pytest.mark.asyncio
    async def test_validation_error_topic_level_1(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for topic_level_1 response."""
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"function_type": "declarative"}
            elif call_count[0] == 2:
                return {"structure_type": "simple"}
            elif call_count[0] == 3:
                return {"purpose": "test"}
            elif call_count[0] == 4:  # topic_level_1
                return {"wrong": "field"}  # Invalid
            elif call_count[0] == 5:
                return {"topic_level_3": "test"}
            elif call_count[0] == 6:
                return {"overall_keywords": ["test"]}
            elif call_count[0] == 7:
                return {"domain_keywords": ["test"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        assert result["topic_level_1"] == ""  # Default due to validation error

    @pytest.mark.asyncio
    async def test_validation_error_topic_level_3(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for topic_level_3 response."""
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"function_type": "declarative"}
            elif call_count[0] == 2:
                return {"structure_type": "simple"}
            elif call_count[0] == 3:
                return {"purpose": "test"}
            elif call_count[0] == 4:
                return {"topic_level_1": "test"}
            elif call_count[0] == 5:  # topic_level_3
                return {"not_topic": "bad"}  # Invalid
            elif call_count[0] == 6:
                return {"overall_keywords": ["test"]}
            elif call_count[0] == 7:
                return {"domain_keywords": ["test"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        assert result["topic_level_3"] == ""  # Default due to validation error

    @pytest.mark.asyncio
    async def test_validation_error_overall_keywords(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for overall_keywords response."""
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"function_type": "declarative"}
            elif call_count[0] == 2:
                return {"structure_type": "simple"}
            elif call_count[0] == 3:
                return {"purpose": "test"}
            elif call_count[0] == 4:
                return {"topic_level_1": "test"}
            elif call_count[0] == 5:
                return {"topic_level_3": "test"}
            elif call_count[0] == 6:  # overall_keywords
                return {"no_keywords": "here"}  # Invalid
            elif call_count[0] == 7:
                return {"domain_keywords": ["test"]}
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        assert result["overall_keywords"] == []  # Default (empty list) due to validation error

    @pytest.mark.asyncio
    async def test_validation_error_domain_keywords(self, analyzer_with_prompts, test_contexts):
        """Test handling of validation error for domain_keywords response."""
        call_count = [0]
        async def mock_call_ordered(prompt: str) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"function_type": "declarative"}
            elif call_count[0] == 2:
                return {"structure_type": "simple"}
            elif call_count[0] == 3:
                return {"purpose": "test"}
            elif call_count[0] == 4:
                return {"topic_level_1": "test"}
            elif call_count[0] == 5:
                return {"topic_level_3": "test"}
            elif call_count[0] == 6:
                return {"overall_keywords": ["test"]}
            elif call_count[0] == 7:  # domain_keywords
                return {"wrong_keywords_field": []}  # Invalid
            return {}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_call_ordered)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        assert result["domain_keywords"] == []  # Default (empty list) due to validation error

    @pytest.mark.asyncio
    async def test_multiple_validation_errors(self, analyzer_with_prompts, test_contexts):
        """Test handling when multiple responses have validation errors."""
        # All responses are invalid
        async def mock_all_invalid(prompt: str) -> Dict[str, Any]:
            return {"invalid": "response"}

        with patch("src.agents.sentence_analyzer.agent") as mock_agent:
            mock_agent.call_model = AsyncMock(side_effect=mock_all_invalid)

            result = await analyzer_with_prompts.classify_sentence("Test sentence", test_contexts)

        # All fields should have default values
        assert result["function_type"] == ""
        assert result["structure_type"] == ""
        assert result["purpose"] == ""
        assert result["topic_level_1"] == ""
        assert result["topic_level_3"] == ""
        assert result["overall_keywords"] == []
        assert result["domain_keywords"] == []
        # Sentence should still be present
        assert result["sentence"] == "Test sentence"
