import os

class ModelConfig:
    """Class to manage model configurations."""
    @staticmethod
    def get_available_providers():
        """Get dictionary of available model providers (chat only, with GPT-4o/4o Mini kept)."""
        return {
            "OpenAI": [
                "GPT-5",
                "GPT-5 Mini",
                "GPT-5 Nano",
                "GPT-4.1",
                "GPT-4.1 Mini",
                "GPT-4o",
                "GPT-4o Mini",
            ],
            "Anthropic": [
                "Claude Sonnet 4.5",
                "Claude Opus 4.1",
                "Claude Opus 4",
                "Claude Sonnet 4",
                "Claude Sonnet 3.7",
            ],
            "DeepSeek": [
                "DeepSeek V3"
            ],
            "Google": [
                "Gemini 2.0 Flash",
                "Gemini 1.5 Pro",
                "Gemini 1.5 Flash",
                "Gemini 1.5 Flash-8B",
            ]
        }

    @staticmethod
    def get_available_models():
        """Get dictionary of available models (chat/text only)."""
        return {
            # Anthropic
            "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929",
            "Claude Opus 4.1": "claude-opus-4-1-20250805",
            "Claude Opus 4": "claude-opus-4-20250514",
            "Claude Sonnet 4": "claude-sonnet-4-20250514",
            "Claude Sonnet 3.7": "claude-3-7-sonnet-20250219",

            # OpenAI (GPT-5 + GPT-4.1 family + GPT-4o)
            "GPT-5": "gpt-5",
            "GPT-5 Mini": "gpt-5-mini",
            "GPT-5 Nano": "gpt-5-nano",
            "GPT-4.1": "gpt-4.1",
            "GPT-4.1 Mini": "gpt-4.1-mini",
            "GPT-4o": "gpt-4o",
            "GPT-4o Mini": "gpt-4o-mini",

            # DeepSeek
            "DeepSeek V3": "deepseek-chat",

            # Google
            "Gemini 2.0 Flash": "gemini-2.0-flash-exp",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.5 Flash-8B": "gemini-1.5-flash-8b",
        }

    @staticmethod
    def get_model_descriptions():
        """Get dictionary of model descriptions."""
        return {
            # Anthropic
            "Claude Sonnet 4.5": "Newest Claude Sonnet, optimized for balanced reasoning and speed.",
            "Claude Opus 4.1": "Latest Opus, strongest Anthropic model for deep analysis.",
            "Claude Opus 4": "High-end reasoning model, excellent for complex workflows.",
            "Claude Sonnet 4": "Next-gen Sonnet, versatile balance of cost and intelligence.",
            "Claude Sonnet 3.7": "Transitional Sonnet, solid performance upgrade over 3.5.",

            # OpenAI
            "GPT-5": "Flagship OpenAI model (2025), state-of-the-art reasoning and text generation.",
            "GPT-5 Mini": "Smaller GPT-5 variant for faster, cheaper tasks.",
            "GPT-5 Nano": "Lightest GPT-5, optimized for very low-latency tasks.",
            "GPT-4.1": "Improved GPT-4 family model, strong balance of power and efficiency.",
            "GPT-4.1 Mini": "Smaller 4.1 variant for quick, cost-effective usage.",
            "GPT-4o": "Versatile flagship model, strong multimodal and reasoning ability.",
            "GPT-4o Mini": "Faster, cheaper 4o variant for focused tasks.",

            # DeepSeek
            "DeepSeek V3": "Mixture-of-Experts (671B total, 37B active per token), strong open-source chat model.",

            # Google
            "Gemini 2.0 Flash": "Latest experimental Gemini, ultra-fast inference.",
            "Gemini 1.5 Pro": "Most capable Gemini in production, advanced reasoning.",
            "Gemini 1.5 Flash": "Fast, efficient Gemini variant for common tasks.",
            "Gemini 1.5 Flash-8B": "Smaller Flash variant (8B), lightweight and fast.",
        }
    
    @staticmethod
    def get_default_model():
        """Get the default model ID."""
        return "claude-3-7-sonnet-20250219"
    
    @staticmethod
    def get_model_name_from_id(model_id):
        """Get model name from model ID."""
        models = ModelConfig.get_available_models()
        for name, mid in models.items():
            if mid == model_id:
                return name
        return None
    
    @staticmethod
    def get_model_id_from_name(model_name):
        """Get model ID from model name."""
        models = ModelConfig.get_available_models()
        return models.get(model_name)
    
    @staticmethod
    def get_models_by_provider(provider_name):
        """Get list of models for a specific provider."""
        providers = ModelConfig.get_available_providers()
        return providers.get(provider_name, [])

    @staticmethod
    def get_model_provider_from_name(model_name):
        """Get provider name for a specific model."""
        providers = ModelConfig.get_available_providers()
        for provider, models in providers.items():
            if model_name in models:
                return provider
        return None
    
    @staticmethod
    def get_model_description(model_name):
        """Get description for a specific model."""
        descriptions = ModelConfig.get_model_descriptions()
        return descriptions.get(model_name, "No description available.")
